//! Reversible, per-REV editing history for authored voxel models.
//!
//! Entries store only the voxel records changed by an operation; the model
//! document is never snapshotted. History is bounded to keep edits on large
//! imported models from retaining unbounded state.

use crate::documents::{ModelVoxel, VoxelModelDocument};
use std::error::Error;
use std::fmt;

/// Maximum number of reversible edits retained across both history stacks.
pub const MAX_MODEL_EDIT_HISTORY: usize = 256;

/// A topology edit targeting one voxel by its current vector index.
///
/// `Remove` captures the removed record when applied. `Replace` supplies the
/// replacement record; the previous record is retained for undo. All fields
/// are full model voxel records, so physical, thermal, and material state is
/// preserved exactly by cloning the record.
#[derive(Clone, Debug)]
pub enum ModelVoxelEdit {
    Insert {
        index: usize,
        voxel: ModelVoxel,
    },
    Remove {
        index: usize,
    },
    Replace {
        index: usize,
        voxel: ModelVoxel,
    },
    /// Applies several edits atomically and stores them as one history entry.
    /// Edits are applied in vector order; undo applies their inverses in
    /// reverse order so index shifts are restored correctly.
    Batch {
        edits: Vec<ModelVoxelEdit>,
    },
}

/// Failure to apply a voxel edit to the current model document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEditError {
    message: String,
}

impl ModelEditError {
    fn index_out_of_bounds(index: usize, len: usize, operation: &'static str) -> Self {
        Self {
            message: format!(
                "cannot {operation} model voxel at index {index}; current voxel count is {len}"
            ),
        }
    }
}

impl fmt::Display for ModelEditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ModelEditError {}

/// Bounded undo/redo history for voxel-model editing operations.
///
/// The stacks store inverse operations, not document snapshots. The default
/// retains at most [`MAX_MODEL_EDIT_HISTORY`] edits. [`Self::with_capacity`]
/// can lower that limit; larger values are clamped to the maximum.
///
/// Callers must route edits to a model through this history, or clear/recreate
/// the history after unrelated document mutations. Indices are intentionally
/// interpreted against the document as it exists when an operation is run.
#[derive(Clone, Debug)]
pub struct ModelEditHistory {
    undo: Vec<ModelVoxelEdit>,
    redo: Vec<ModelVoxelEdit>,
    capacity: usize,
}

impl Default for ModelEditHistory {
    fn default() -> Self {
        Self::with_capacity(MAX_MODEL_EDIT_HISTORY)
    }
}

impl ModelEditHistory {
    /// Creates an empty history retaining at most `capacity` edits.
    ///
    /// The requested capacity is clamped to [`MAX_MODEL_EDIT_HISTORY`]. A
    /// zero capacity still applies committed edits but retains none for undo.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            capacity: capacity.min(MAX_MODEL_EDIT_HISTORY),
        }
    }

    /// Applies an edit and records its inverse. A failed edit leaves both the
    /// document and history stacks unchanged.
    pub fn commit(
        &mut self,
        model: &mut VoxelModelDocument,
        edit: ModelVoxelEdit,
    ) -> Result<(), ModelEditError> {
        let inverse = apply_and_invert(model, &edit)?;
        self.redo.clear();
        self.push_bounded_undo(inverse);
        Ok(())
    }

    /// Reverts the most recent committed edit. Returns `Ok(false)` when there
    /// is no undo entry. Invalid state/index errors leave history unchanged.
    pub fn undo(&mut self, model: &mut VoxelModelDocument) -> Result<bool, ModelEditError> {
        let Some(edit) = self.undo.last() else {
            return Ok(false);
        };
        let forward = apply_and_invert(model, edit)?;
        self.undo.pop();
        self.redo.push(forward);
        Ok(true)
    }

    /// Reapplies the most recently undone edit. Returns `Ok(false)` when there
    /// is no redo entry. Invalid state/index errors leave history unchanged.
    pub fn redo(&mut self, model: &mut VoxelModelDocument) -> Result<bool, ModelEditError> {
        let Some(edit) = self.redo.last() else {
            return Ok(false);
        };
        let inverse = apply_and_invert(model, edit)?;
        self.redo.pop();
        self.push_bounded_undo(inverse);
        Ok(true)
    }

    /// Discards all retained undo and redo operations.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn push_bounded_undo(&mut self, edit: ModelVoxelEdit) {
        if self.capacity == 0 {
            return;
        }
        if self.undo.len() == self.capacity {
            self.undo.remove(0);
        }
        self.undo.push(edit);
    }
}

fn apply_and_invert(
    model: &mut VoxelModelDocument,
    edit: &ModelVoxelEdit,
) -> Result<ModelVoxelEdit, ModelEditError> {
    let len = model.voxels.len();
    match edit {
        ModelVoxelEdit::Insert { index, voxel } => {
            if *index > len {
                return Err(ModelEditError::index_out_of_bounds(*index, len, "insert"));
            }
            model.voxels.insert(*index, voxel.clone());
            Ok(ModelVoxelEdit::Remove { index: *index })
        }
        ModelVoxelEdit::Remove { index } => {
            if *index >= len {
                return Err(ModelEditError::index_out_of_bounds(*index, len, "remove"));
            }
            let removed = model.voxels.remove(*index);
            Ok(ModelVoxelEdit::Insert {
                index: *index,
                voxel: removed,
            })
        }
        ModelVoxelEdit::Replace { index, voxel } => {
            if *index >= len {
                return Err(ModelEditError::index_out_of_bounds(*index, len, "replace"));
            }
            let previous = std::mem::replace(&mut model.voxels[*index], voxel.clone());
            Ok(ModelVoxelEdit::Replace {
                index: *index,
                voxel: previous,
            })
        }
        ModelVoxelEdit::Batch { edits } => {
            let mut inverses = Vec::with_capacity(edits.len());
            for nested in edits {
                match apply_and_invert(model, nested) {
                    Ok(inverse) => inverses.push(inverse),
                    Err(error) => {
                        // Each inverse was produced from a successful edit at
                        // the exact preceding state. Replaying them backward
                        // therefore restores both records and shifted indices.
                        for inverse in inverses.iter().rev() {
                            let rollback = apply_and_invert(model, inverse);
                            debug_assert!(
                                rollback.is_ok(),
                                "generated model edit inverse must be applicable"
                            );
                        }
                        return Err(error);
                    }
                }
            }
            inverses.reverse();
            Ok(ModelVoxelEdit::Batch { edits: inverses })
        }
    }
}
