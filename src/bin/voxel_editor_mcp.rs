use glam::{DVec3, Quat, Vec3};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};
use voxel_physics_lab::{
    documents::{
        generate_auto_bone_weights, load_document, save_document, BoneKeyframeDocument,
        ConstraintPartDocument, EditorDocument, ManualBondDocument, ManualBondEndpoint,
        ModelInstanceDocument, ModelVoxel, SceneDocument,
        SkeletonDocument, VoxelModelDocument,
    },
    material::Material,
    mesh_import::{import_mesh, VoxelizationMode, VoxelizationOptions},
    creature_inspector::{inspect as inspect_creature, CreatureFile},
    model_export::export_obj,
    procedural_animation::{author_walk, WalkOptions},
    scene::{SceneBuilder, SceneConfig},
    scene_chemistry::ChemistrySettings,
    voxel::Composition,
};
const LEGACY_PROTOCOL_VERSION: &str = "2025-11-25";
const MODERN_PROTOCOL_VERSION: &str = "2026-07-28";
const SETTINGS_FILE: &str = ".voxel_editor_settings.json";
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Deserialize)]
struct Settings {
    version: u32,
    #[serde(default)]
    mcp_enabled: bool,
}

struct App {
    root: PathBuf,
}

fn main() {
    let root_arg = parse_root_arg();
    let root = root_arg
        .and_then(|path| fs::canonicalize(path).map_err(|e| format!("resolve project root: {e}")))
        .unwrap_or_else(|error| {
            // Diagnostics go to stderr; stdout is reserved for MCP messages.
            eprintln!("voxel-editor-mcp: {error}");
            std::process::exit(2);
        });
    if !root.is_dir() {
        eprintln!("voxel-editor-mcp: project root is not a directory");
        std::process::exit(2);
    }
    if let (_, Some(error)) = read_settings(&root) {
        eprintln!("voxel-editor-mcp: {error}");
    }
    let app = App { root };
    if let Err(error) = serve(app) {
        eprintln!("voxel-editor-mcp: {error}");
        std::process::exit(1);
    }
}

fn parse_root_arg() -> Result<PathBuf, String> {
    let mut args = env::args_os().skip(1);
    let mut root = None;
    while let Some(arg) = args.next() {
        if arg == "--project-root" {
            if root.is_some() {
                return Err("--project-root may be specified only once".into());
            }
            root = Some(PathBuf::from(
                args.next().ok_or("--project-root needs a path")?,
            ));
        } else {
            return Err(format!("unknown argument: {}", arg.to_string_lossy()));
        }
    }
    Ok(root.unwrap_or(env::current_dir().map_err(|e| format!("read current directory: {e}"))?))
}

fn read_settings(root: &Path) -> (bool, Option<String>) {
    let path = root.join(SETTINGS_FILE);
    let contents = match fs::read(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return (true, None),
        Err(error) => return (false, Some(format!("cannot read settings: {error}"))),
    };
    match serde_json::from_slice::<Settings>(&contents) {
        Ok(settings) if settings.version == 1 => (settings.mcp_enabled, None),
        Ok(settings) => (
            false,
            Some(format!("unsupported settings version {}", settings.version)),
        ),
        Err(error) => (false, Some(format!("invalid settings JSON: {error}"))),
    }
}

fn serve(app: App) -> Result<(), String> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().split(b'\n') {
        let line = line.map_err(|e| format!("read stdin: {e}"))?;
        if line.is_empty() {
            continue;
        }
        if line.len() > MAX_LINE_BYTES {
            write_message(
                &mut stdout,
                &json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"request too large"}}),
            )?;
            continue;
        }
        let request: Value = match serde_json::from_slice(&line) {
            Ok(value) => value,
            Err(error) => {
                write_message(
                    &mut stdout,
                    &json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":error.to_string()}}),
                )?;
                continue;
            }
        };
        if request.get("jsonrpc").and_then(Value::as_str) != Some("2.0")
            || request.get("method").and_then(Value::as_str).is_none()
        {
            if let Some(id) = request.get("id") {
                write_message(
                    &mut stdout,
                    &json!({"jsonrpc":"2.0","id":id,"error":{"code":-32600,"message":"invalid JSON-RPC request"}}),
                )?;
            }
            continue;
        }
        if request.get("id").is_none() {
            // notifications, including notifications/initialized
            continue;
        }
        let id = request["id"].clone();
        let method = request["method"].as_str().unwrap_or_default();
        let modern_version = request
            .pointer("/params/_meta/io.modelcontextprotocol~1protocolVersion")
            .and_then(Value::as_str);
        let modern = modern_version.is_some();
        let result = if modern {
            let version = modern_version.unwrap_or_default();
            let capabilities = request
                .pointer("/params/_meta/io.modelcontextprotocol~1clientCapabilities")
                .and_then(Value::as_object);
            if version != MODERN_PROTOCOL_VERSION {
                Err((
                    -32022,
                    format!("unsupported protocol version {version}; supported: {MODERN_PROTOCOL_VERSION}"),
                ))
            } else if capabilities.is_none() {
                Err((
                    -32602,
                    "modern MCP requests require clientCapabilities metadata".into(),
                ))
            } else {
                dispatch_method(&app, method, &request["params"], true)
            }
        } else if method == "server/discover" {
            Err((
                -32602,
                "server/discover requires modern protocol metadata".into(),
            ))
        } else {
            dispatch_method(&app, method, &request["params"], false)
        };
        let mut response = match result {
            Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
            Err((code, message)) => {
                json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
            }
        };
        if let Some(error) = response.get_mut("error") {
            if error.get("code").and_then(Value::as_i64) == Some(-32022) {
                error["data"] = json!({
                    "supported":[MODERN_PROTOCOL_VERSION],
                    "requested":modern_version.unwrap_or_default()
                });
            }
        }
        if modern {
            response["_meta"] = json!({"io.modelcontextprotocol/serverInfo": {
                "name":"voxel-editor-local",
                "version":env!("CARGO_PKG_VERSION")
            }});
        }
        write_message(&mut stdout, &response)?;
    }
    Ok(())
}

fn dispatch_method(
    app: &App,
    method: &str,
    params: &Value,
    modern: bool,
) -> Result<Value, (i64, String)> {
    match method {
        "server/discover" if modern => Ok(json!({
            "supportedVersions":[MODERN_PROTOCOL_VERSION],
            "capabilities":{"tools":{"listChanged":false}},
            "instructions":"Use voxel tools to create, inspect, edit, animate, import, export, and arrange saved native assets beneath the configured project root. File mutation tools create new files and refuse overwrite. Read-only live telemetry tools query the running native engine; they do not edit live state.",
            "ttlMs":60000,
            "cacheScope":"private",
            "resultType":"complete"
        })),
        "initialize" if !modern => initialize(params),
        "initialize" => Err((
            -32601,
            "initialize belongs to the legacy MCP protocol".into(),
        )),
        "ping" => Ok(if modern {
            json!({"resultType":"complete"})
        } else {
            json!({})
        }),
        "tools/list" => {
            let (enabled, _) = read_settings(&app.root);
            let mut result = json!({"tools": if enabled { tool_definitions() } else { vec![] }});
            if modern {
                result["ttlMs"] = json!(60000);
                result["cacheScope"] = json!("private");
                result["resultType"] = json!("complete");
            }
            Ok(result)
        }
        "tools/call" => {
            let mut result = call_tool(app, params)?;
            if modern {
                result["resultType"] = json!("complete");
            }
            Ok(result)
        }
        _ => Err((-32601, format!("method not found: {method}"))),
    }
}

fn initialize(params: &Value) -> Result<Value, (i64, String)> {
    let requested = params["protocolVersion"]
        .as_str()
        .unwrap_or(LEGACY_PROTOCOL_VERSION);
    if requested != LEGACY_PROTOCOL_VERSION {
        return Err((
            -32602,
            format!(
                "unsupported legacy protocol version {requested}; supported version is {LEGACY_PROTOCOL_VERSION}"
            ),
        ));
    }
    Ok(json!({
        "protocolVersion": LEGACY_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": false}},
        "serverInfo": {"name":"voxel-editor-local","version":env!("CARGO_PKG_VERSION")},
        "instructions":"File tools operate on saved .vxm/.vxscene/.obj/.glb files beneath the configured project root; mutations write new files and refuse overwrite. get_runtime_status, read_runtime_metrics, get_runtime_objects and get_run_panel are read-only queries to the running native engine and do not edit live state. get_scene_chemistry, set_scene_chemistry and inspect_creatures work on the scene files beside a .vxscene."
    }))
}

fn tool_definitions() -> Vec<Value> {
    vec![
        tool_definition("list_documents", "List saved .vxm and .vxscene assets beneath the project root.", json!({"type":"object","properties":{},"additionalProperties":false}), true),
        tool_definition("batch_actions", "Run up to 64 validated MCP tool calls sequentially. All tool names, arguments, and IDs are validated before the first action runs. Earlier successful file writes remain if a later action fails; this is not a transaction. Telemetry waits share a 10-second total budget.", json!({"type":"object","properties":{"actions":{"type":"array","minItems":1,"maxItems":64,"items":{"type":"object","properties":{"id":{"type":"string","minLength":1,"maxLength":64},"name":{"type":"string","minLength":1},"arguments":{"type":"object"}},"required":["name","arguments"],"additionalProperties":false}},"stop_on_error":{"type":"boolean","default":true}},"required":["actions"],"additionalProperties":false}), false),
        tool_definition("get_runtime_status", "Read the running native engine's status, session identity, and telemetry freshness. Read-only; errors clearly when the runtime is absent or stale.", json!({"type":"object","properties":{},"additionalProperties":false}), true),
        tool_definition("read_runtime_metrics", "Read a bounded batch of live engine telemetry after a sequence cursor. Returns explicit freshness, session identity, and sequence-gap markers. Read-only.", json!({"type":"object","properties":{"after_sequence":{"type":"integer","minimum":0,"default":0},"max_samples":{"type":"integer","minimum":1,"maximum":64,"default":16},"wait_ms":{"type":"integer","minimum":0,"maximum":10000,"default":0},"topic":{"type":"string","enum":["app","physics","event","objects"]}},"additionalProperties":false}), true),
        tool_definition("get_runtime_objects", "Read a snapshot of objects currently known to the native runtime. Read-only.", json!({"type":"object","properties":{},"additionalProperties":false}), true),
        tool_definition("inspect_document", "Validate and summarize a .vxm or .vxscene, including its materials, groups, constraints, skeleton, keyframes, and referenced assets.", json!({"type":"object","properties":{"path":{"type":"string","description":"Project-relative document path"}},"required":["path"],"additionalProperties":false}), true),
        tool_definition("voxelize_mesh", "Convert an OBJ or GLB mesh inside the project root into a native .vxm shell or filled voxel model.", json!({"type":"object","properties":{"source_path":{"type":"string"},"output_path":{"type":"string"},"model_name":{"type":"string"},"scale":{"type":"number","minimum":0.01,"maximum":100},"mode":{"type":"string","enum":["shell","filled"]}},"required":["source_path","output_path"],"additionalProperties":false}), false),
        tool_definition("export_model_obj", "Export exposed faces of a saved .vxm as a watertight OBJ mesh. OBJ coordinates are metres; the native model remains unchanged.", json!({"type":"object","properties":{"source_path":{"type":"string"},"output_path":{"type":"string"}},"required":["source_path","output_path"],"additionalProperties":false}), false),
        tool_definition("create_model", "Create a valid editable .vxm starter model. Refuses to overwrite existing files.", json!({"type":"object","properties":{"output_path":{"type":"string"},"model_name":{"type":"string"}},"required":["output_path"],"additionalProperties":false}), false),
        tool_definition("create_scene", "Create an empty authored .vxscene. Refuses to overwrite existing files.", json!({"type":"object","properties":{"output_path":{"type":"string"},"scene_name":{"type":"string"}},"required":["output_path"],"additionalProperties":false}), false),
        tool_definition("edit_model_skeleton", "Author bones, clips, keyframes, and automatic or corrected REV weights. Each operation writes and validates a new .vxm.", json!({"type":"object","properties":{"source_path":{"type":"string"},"output_path":{"type":"string"},"operation":{"type":"string","enum":["create_root","add_bone","create_clip","set_keyframe","auto_weights","adjust_weight"]},"name":{"type":"string"},"parent_bone":{"type":"integer","minimum":0},"bone_index":{"type":"integer","minimum":0},"clip_index":{"type":"integer","minimum":0},"voxel_index":{"type":"integer","minimum":0},"position_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"time_s":{"type":"number","minimum":0},"duration_s":{"type":"number","exclusiveMinimum":0},"translation_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"rotation_xyzw":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"max_influences":{"type":"integer","minimum":1,"maximum":4},"weight_delta":{"type":"number"}},"required":["source_path","output_path","operation"],"additionalProperties":false}), false),
        tool_definition("author_animation", "Create a rig and clip in one MCP call. Positions and keyframe translations are parent-local metres; list bones after their parent (root is index 0). Rotations are quaternions [x,y,z,w]. Adds automatic REV weights and writes a new validated .vxm.", json!({"type":"object","properties":{"source_path":{"type":"string"},"output_path":{"type":"string"},"root_name":{"type":"string"},"root_position_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"bones":{"type":"array","minItems":1,"items":{"type":"object","properties":{"name":{"type":"string"},"parent_bone":{"type":"integer","minimum":0},"position_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}},"required":["name","parent_bone","position_m"],"additionalProperties":false}},"clip_name":{"type":"string"},"duration_s":{"type":"number","exclusiveMinimum":0},"keyframes":{"type":"array","minItems":1,"items":{"type":"object","properties":{"bone_index":{"type":"integer","minimum":0},"time_s":{"type":"number","minimum":0},"translation_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"rotation_xyzw":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}},"required":["bone_index","time_s","translation_m","rotation_xyzw"],"additionalProperties":false}},"max_influences":{"type":"integer","minimum":1,"maximum":4}},"required":["source_path","output_path","root_name","root_position_m","bones","clip_name","duration_s","keyframes"],"additionalProperties":false}), false),
        tool_definition("author_procedural_gait", "Build a smooth in-place IK walk from a raised body and 2-16 distinct leg contacts. Infers the real connected geometry, rigidly binds body and limb segments, preserves bone lengths, and generates a closed wave gait with ground stance and lifted return strokes. Source is preserved.", json!({
            "type":"object","properties":{
                "source_path":{"type":"string","description":"Project-relative rig-free .vxm."},
                "output_path":{"type":"string","description":"New .vxm path; existing files are never overwritten."},
                "clip_name":{"type":"string","default":"Walk"},
                "duration_s":{"type":"number","minimum":0.1,"maximum":10,"default":1.8,"description":"Duration of one closed in-place walk cycle."},
                "up_axis":{"type":"string","enum":["auto","x","y","z"],"default":"auto","description":"Vertical model axis, or infer from foot contacts."},
                "forward_direction":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3,"description":"Horizontal walking direction, normalized internally. Defaults to +Y for Z-up."},
                "stride_m":{"type":"number","minimum":0,"maximum":0.5,"default":0.025,"description":"Backward travel during stance and forward travel during return."},
                "lift_m":{"type":"number","minimum":0,"maximum":0.5,"default":0.014,"description":"Maximum foot clearance above the ground."},
                "stance_fraction":{"type":"number","minimum":0.55,"maximum":0.95,"default":0.80,"description":"Fraction of each foot cycle spent on the ground."},
                "sample_fps":{"type":"number","minimum":15,"maximum":120,"default":120,"description":"Pose bake rate; runtime playback interpolates these keys."}
            },"required":["source_path","output_path"],"additionalProperties":false
        }), false),
        tool_definition("sample_animation", "Sample interpolated local bone poses and global joints from a clip at a time in seconds.", json!({"type":"object","properties":{"path":{"type":"string"},"clip_index":{"type":"integer","minimum":0},"time_s":{"type":"number","minimum":0}},"required":["path","clip_index","time_s"],"additionalProperties":false}), true),
        tool_definition("edit_voxel_model", "Edit a .vxm on its REV grid: add, move, duplicate or remove one REV; stamp a block, sphere, cylinder or frame; add a built-in material preset or assign a material already stored in the model. Every change writes a validated new file.", json!({"type":"object","properties":{"source_path":{"type":"string"},"output_path":{"type":"string"},"operation":{"type":"string","enum":["add_voxel","remove_voxel","move_voxel","duplicate_voxel","add_material","assign_material","stamp_shape"]},"voxel_index":{"type":"integer","minimum":0},"cell":{"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":3},"min_cell":{"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":3},"max_cell":{"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":3},"shape":{"type":"string","enum":["block","sphere","cylinder","frame"]},"material_name":{"type":"string"}},"required":["source_path","output_path","operation"],"additionalProperties":false}), false),
        tool_definition("edit_scene", "Add, move, duplicate or remove .vxm instances in a saved .vxscene, preserving project-relative references and returning resulting transforms. Also authors explicit bonds between REVs of two instances (add_manual_bond, set_manual_bond, remove_manual_bond; rest_length_m defaults to 0.01, pretension_n to 0) and make_unique, which copies one instance's .vxm to new_asset_path (never overwriting) and points only that instance at the copy.", json!({"type":"object","properties":{"source_path":{"type":"string"},"output_path":{"type":"string"},"operation":{"type":"string","enum":["add_instance","move_instance","duplicate_instance","remove_instance","add_manual_bond","set_manual_bond","remove_manual_bond","make_unique"]},"instance_index":{"type":"integer","minimum":0},"bond_index":{"type":"integer","minimum":0},"instance_a":{"type":"integer","minimum":0},"voxel_a":{"type":"integer","minimum":0},"instance_b":{"type":"integer","minimum":0},"voxel_b":{"type":"integer","minimum":0},"rest_length_m":{"type":"number","exclusiveMinimum":0},"pretension_n":{"type":"number"},"new_asset_path":{"type":"string"},"asset_path":{"type":"string"},"translation_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"rotation_xyzw":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"scale":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"delta_cells":{"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":3}},"required":["source_path","output_path","operation"],"additionalProperties":false}), false),
        tool_definition("get_run_panel", "Read the editor's run panel: state (Run, Pause, Failed), measured world speed, backlog, worker cost per 10 ms of physics (latest, p50, p95, max over the last 128 batches), active mechanical REV count and solver mode (cell rotations or translation only). The editor publishes the same values it shows in its Run tab. Read-only; errors when no editor is running or nothing was published yet.", json!({"type":"object","properties":{},"additionalProperties":false}), true),
        tool_definition("get_scene_chemistry", "Read a scene's chemistry settings (reactions on or off, interval in simulated seconds, recipe file) from <scene>.chemistry.json beside the .vxscene and check that the recipe file still loads against the scene's material table. A scene without the file reports the defaults (reactions off, built-in recipes). Read-only.", json!({"type":"object","properties":{"scene_path":{"type":"string","description":"Project-relative .vxscene."}},"required":["scene_path"],"additionalProperties":false}), true),
        tool_definition("set_scene_chemistry", "Change a scene's chemistry settings, the same operation as the editor's Chemistry section: enabled turns reactions on or off, interval_s sets the simulated time between reaction updates (0 = every step), recipe_path loads a recipe JSON (see recipes/wood_rot.json; it replaces the built-in recipes) and use_builtin_recipes returns to the built-in ones. The recipe is parsed and resolved against the scene's materials and the result is applied to a scratch scene before anything is written. Updates <scene>.chemistry.json in place (atomically); it never edits the .vxscene. A running editor reads the file when it opens the scene.", json!({"type":"object","properties":{"scene_path":{"type":"string"},"enabled":{"type":"boolean"},"interval_s":{"type":"number","minimum":0,"maximum":31557600},"recipe_path":{"type":"string","description":"Project-relative recipe .json."},"use_builtin_recipes":{"type":"boolean"}},"required":["scene_path"],"additionalProperties":false}), false),
        tool_definition("inspect_creatures", "Read-only creature inspector: for each creature listed in <scene>.creatures.json beside the .vxscene, report the resource ledger (stored energy, work, heat), every sensor channel with its unit and value, the sensor read cost counters and the network size and cost counters. A scene without that file has no creatures and reports an empty list. Inspecting changes nothing; a creature the editor does not drive shows its birth ledger.", json!({"type":"object","properties":{"scene_path":{"type":"string","description":"Project-relative .vxscene."}},"required":["scene_path"],"additionalProperties":false}), true),
    ]
}

fn tool_definition(name: &str, description: &str, input_schema: Value, read_only: bool) -> Value {
    json!({
        "name":name,
        "description":description,
        "inputSchema":input_schema,
        "annotations":{"readOnlyHint":read_only,"openWorldHint":false}
    })
}

fn call_tool(app: &App, params: &Value) -> Result<Value, (i64, String)> {
    let name = params["name"]
        .as_str()
        .ok_or((-32602, "tools/call requires a tool name".into()))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let (enabled, settings_error) = read_settings(&app.root);
    if !enabled {
        let reason = settings_error
            .as_deref()
            .unwrap_or("MCP is disabled in .voxel_editor_settings.json");
        return Ok(tool_error(reason));
    }
    if let Err(error) = validate_tool_arguments(name, &args) {
        return Ok(tool_error(&error));
    }
    if name == "batch_actions" {
        if let Err(error) = validate_batch_actions(&args) {
            return Ok(tool_error(&error));
        }
        return Ok(batch_actions(app, &args));
    }
    Ok(call_tool_validated(app, name, &args))
}

fn validate_batch_actions(args: &Value) -> Result<(), String> {
    let actions = args["actions"]
        .as_array()
        .ok_or("actions must be an array")?;
    if actions.is_empty() || actions.len() > 64 {
        return Err("actions must contain between 1 and 64 items".into());
    }
    let mut ids = HashSet::new();
    let mut total_wait_ms = 0_u64;
    for (index, action) in actions.iter().enumerate() {
        let object = action.as_object().ok_or_else(|| format!("actions[{index}] must be an object"))?;
        if let Some(extra) = object.keys().find(|key| !["id", "name", "arguments"].contains(&key.as_str())) {
            return Err(format!("actions[{index}] has unknown field {extra:?}"));
        }
        if let Some(id) = object.get("id") {
            let id = id.as_str().filter(|id| !id.trim().is_empty() && id.len() <= 64)
                .ok_or_else(|| format!("actions[{index}].id must be a non-empty string of at most 64 bytes"))?;
            if !ids.insert(id) {
                return Err(format!("duplicate action id {id:?}"));
            }
        }
        let name = object.get("name").and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| format!("actions[{index}].name must be a non-empty string"))?;
        if name == "batch_actions" {
            return Err("nested batch_actions are not supported".into());
        }
        let arguments = object.get("arguments").ok_or_else(|| format!("actions[{index}].arguments is required"))?;
        validate_tool_arguments(name, arguments).map_err(|error| format!("actions[{index}] ({name}): {error}"))?;
        if name == "read_runtime_metrics" {
            total_wait_ms += arguments.get("wait_ms").and_then(Value::as_u64).unwrap_or(0);
            if total_wait_ms > 10_000 {
                return Err("combined read_runtime_metrics wait_ms exceeds the 10,000 ms batch limit".into());
            }
        }
    }
    Ok(())
}

fn call_tool_validated(app: &App, name: &str, args: &Value) -> Value {
    let result = match name {
        "list_documents" => list_documents(app),
        "get_runtime_status" => runtime_query(app, "status", json!({})),
        "read_runtime_metrics" => runtime_query(app, "metrics", args.clone()),
        "get_runtime_objects" => runtime_query(app, "objects", json!({})),
        "inspect_document" => inspect_document(app, &args),
        "voxelize_mesh" => voxelize_mesh(app, &args),
        "export_model_obj" => export_model_obj(app, &args),
        "create_model" => create_model(app, &args),
        "create_scene" => create_scene(app, &args),
        "edit_model_skeleton" => edit_model_skeleton(app, &args),
        "author_animation" => author_animation(app, &args),
        "author_procedural_gait" => author_procedural_gait(app, &args),
        "sample_animation" => sample_animation(app, &args),
        "edit_voxel_model" => edit_voxel_model(app, &args),
        "edit_scene" => edit_scene(app, &args),
        "get_run_panel" => get_run_panel(app),
        "get_scene_chemistry" => get_scene_chemistry(app, &args),
        "set_scene_chemistry" => set_scene_chemistry(app, &args),
        "inspect_creatures" => inspect_creatures(app, &args),
        _ => return tool_error(&format!("unknown tool: {name}")),
    };
    match result {
        Ok(value) => tool_result(value),
        Err(error) => tool_error(&error),
    }
}

fn batch_actions(app: &App, args: &Value) -> Value {
    let actions = args["actions"].as_array().expect("batch schema validated");
    let stop_on_error = args["stop_on_error"].as_bool().unwrap_or(true);
    let mut results = Vec::with_capacity(actions.len());
    let mut failed = false;
    for (index, action) in actions.iter().enumerate() {
        let id = action.get("id").cloned().unwrap_or(Value::Null);
        let name = action["name"].as_str().expect("batch prevalidated");
        let result = call_tool_validated(app, name, &action["arguments"]);
        if result["isError"] == true {
            failed = true;
            results.push(json!({"index":index,"id":id,"name":name,"status":"error","error":result["content"][0]["text"]}));
            if stop_on_error {
                for (skipped_index, skipped) in actions.iter().enumerate().skip(index + 1) {
                    results.push(json!({"index":skipped_index,"id":skipped.get("id").cloned().unwrap_or(Value::Null),"name":skipped["name"],"status":"skipped","reason":"stop_on_error"}));
                }
                break;
            }
        } else {
            results.push(json!({"index":index,"id":id,"name":name,"status":"success","result":result["structuredContent"]}));
        }
    }
    compact_batch_results(&mut results);
    let response = json!({"action_count":actions.len(),"executed_count":results.iter().filter(|r| r["status"] != "skipped").count(),"success_count":results.iter().filter(|r| r["status"] == "success").count(),"error_count":results.iter().filter(|r| r["status"] == "error").count(),"skipped_count":results.iter().filter(|r| r["status"] == "skipped").count(),"results":results});
    let mut output = tool_result(response);
    output["isError"] = json!(failed);
    output
}

fn compact_batch_results(results: &mut [Value]) {
    const MAX_RESPONSE_BYTES: usize = 1_000_000;
    loop {
        let mut output = tool_result(json!({"results":results}));
        if output.to_string().len() <= MAX_RESPONSE_BYTES {
            break;
        }
        let Some(result) = results.iter_mut().rev().find(|result| {
            result["status"] == "success" && result["result_omitted"] != true
        }) else {
            break;
        };
        let original = std::mem::replace(&mut result["result"], Value::Null);
        let original_bytes = original.to_string().len();
        let path = original.get("path").cloned();
        result["result"] = json!({"result_omitted":true,"original_bytes":original_bytes,"path":path,"reason":"batch response size limit"});
        result["result_omitted"] = json!(true);
    }
}

fn runtime_query(app: &App, command: &str, mut request: Value) -> Result<Value, String> {
    let object = request
        .as_object_mut()
        .ok_or("runtime request arguments must be an object")?;
    object.insert("command".into(), json!(command));
    voxel_physics_lab::runtime_metrics::query_runtime(&app.root, &request)
}

/// The run panel the editor last published (the values of its Run tab).
fn get_run_panel(app: &App) -> Result<Value, String> {
    let response = runtime_query(app, "latest", json!({"topic":"run_panel"}))?;
    if response["ok"] != true {
        return Err(format!(
            "runtime refused the run panel request: {}",
            response["error"].as_str().unwrap_or("unknown error")
        ));
    }
    if response["data"].is_null() {
        return Err("the editor has not published a run panel yet".into());
    }
    Ok(json!({
        "run_panel": response["data"],
        "sequence": response["sequence"],
        "age_ms": response["age_ms"],
        "session_id": response["session_id"],
    }))
}

fn scene_document_path(app: &App, args: &Value) -> Result<PathBuf, String> {
    let path = existing_path(app, required_string(args, "scene_path")?)?;
    if extension(&path).as_deref() != Some("vxscene") {
        return Err("scene_path must refer to a .vxscene file".into());
    }
    Ok(path)
}

fn chemistry_result(
    app: &App,
    scene_path: &Path,
    settings: &ChemistrySettings,
    recipe_count: Result<Option<usize>, String>,
) -> Value {
    let (count, error) = match recipe_count {
        Ok(count) => (count, None),
        Err(error) => (None, Some(error)),
    };
    json!({
        "scene_path": relative(&app.root, scene_path),
        "settings_file": relative(&app.root, &ChemistrySettings::sidecar_path(scene_path)),
        "settings": settings.to_json(),
        "summary": settings.summary(),
        "recipe_count": count,
        "recipe_error": error,
    })
}

fn get_scene_chemistry(app: &App, args: &Value) -> Result<Value, String> {
    let scene_path = scene_document_path(app, args)?;
    let document: SceneDocument = load_document(&scene_path)?;
    let scene = document.build_scene(&scene_path)?;
    let settings = ChemistrySettings::load(&scene_path)?;
    let recipes = settings.check_recipe(Some(&scene_path), &scene.materials);
    Ok(chemistry_result(app, &scene_path, &settings, recipes))
}

fn set_scene_chemistry(app: &App, args: &Value) -> Result<Value, String> {
    let scene_path = scene_document_path(app, args)?;
    let enabled = args.get("enabled").and_then(Value::as_bool);
    let interval_s = args.get("interval_s").and_then(Value::as_f64);
    let recipe = args.get("recipe_path").and_then(Value::as_str);
    let builtin = args.get("use_builtin_recipes").and_then(Value::as_bool) == Some(true);
    if enabled.is_none() && interval_s.is_none() && recipe.is_none() && !builtin {
        return Err("give at least one of enabled, interval_s, recipe_path, use_builtin_recipes".into());
    }
    if recipe.is_some() && builtin {
        return Err("recipe_path and use_builtin_recipes are mutually exclusive".into());
    }
    let document: SceneDocument = load_document(&scene_path)?;
    let mut scratch = document.build_scene(&scene_path)?;
    let mut settings = ChemistrySettings::load(&scene_path)?;
    if let Some(enabled) = enabled {
        settings.set_enabled(enabled);
    }
    if let Some(interval_s) = interval_s {
        settings.set_interval(interval_s)?;
    }
    if builtin {
        settings.clear_recipe();
    }
    if let Some(recipe) = recipe {
        let recipe_file = existing_path(app, recipe)?;
        if extension(&recipe_file).as_deref() != Some("json") {
            return Err("recipe_path must refer to a .json recipe file".into());
        }
        settings.set_recipe_file(&scene_path, &recipe_file, &scratch.materials)?;
    }
    // Prove the whole result works on a scene before it is written.
    settings.apply(&mut scratch, Some(&scene_path))?;
    let recipes = settings.check_recipe(Some(&scene_path), &scratch.materials);
    settings.save(&scene_path)?;
    Ok(chemistry_result(app, &scene_path, &settings, recipes))
}

fn inspect_creatures(app: &App, args: &Value) -> Result<Value, String> {
    let scene_path = scene_document_path(app, args)?;
    let document: SceneDocument = load_document(&scene_path)?;
    let scene = document.build_scene(&scene_path)?;
    let Some(file) = CreatureFile::load(&scene_path)? else {
        return Ok(json!({
            "scene_path": relative(&app.root, &scene_path),
            "creature_count": 0,
            "creatures": [],
            "note": "no creature file beside the scene; the scene has no creature",
        }));
    };
    let creatures = file.build(&scene)?;
    let inspections = creatures
        .iter()
        .map(|named| inspect_creature(&named.name, &named.creature, &scene, None).map(|view| view.to_json()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "scene_path": relative(&app.root, &scene_path),
        "creature_count": inspections.len(),
        "creatures": inspections,
        "note": "read-only; the editor does not drive creatures, so ledgers show the birth state and the network is not running",
    }))
}

fn validate_tool_arguments(name: &str, args: &Value) -> Result<(), String> {
    let Some(definition) = tool_definitions()
        .into_iter()
        .find(|definition| definition["name"].as_str() == Some(name))
    else {
        return Err(format!("unknown tool: {name}"));
    };
    validate_schema_value(&definition["inputSchema"], args, "arguments")
}

fn validate_schema_value(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    let kind = schema["type"].as_str().unwrap_or_default();
    let valid_type = match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some(),
        "number" => value.as_f64().is_some_and(f64::is_finite),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => true,
    };
    if !valid_type {
        return Err(format!("{path} must be a {kind}"));
    }
    if let Some(choices) = schema["enum"].as_array() {
        if !choices.contains(value) {
            return Err(format!("{path} must be one of {choices:?}"));
        }
    }
    if let Some(text) = value.as_str() {
        if schema["minLength"].as_u64().is_some_and(|min| text.len() < min as usize)
            || schema["maxLength"].as_u64().is_some_and(|max| text.len() > max as usize)
        {
            return Err(format!("{path} has an invalid string length"));
        }
    }
    if let Some(number) = value.as_f64() {
        if schema["minimum"].as_f64().is_some_and(|min| number < min) {
            return Err(format!("{path} is below its minimum"));
        }
        if schema["maximum"].as_f64().is_some_and(|max| number > max) {
            return Err(format!("{path} is above its maximum"));
        }
        if schema["exclusiveMinimum"].as_f64().is_some_and(|min| number <= min) {
            return Err(format!("{path} must be greater than its minimum"));
        }
    }
    if let Some(values) = value.as_array() {
        if schema["minItems"].as_u64().is_some_and(|min| values.len() < min as usize)
            || schema["maxItems"].as_u64().is_some_and(|max| values.len() > max as usize)
        {
            return Err(format!("{path} has an invalid array length"));
        }
        if schema.get("items").is_some() {
            for (index, item) in values.iter().enumerate() {
                validate_schema_value(&schema["items"], item, &format!("{path}[{index}]"))?;
            }
        }
    }
    if let Some(object) = value.as_object() {
        let properties = schema["properties"].as_object();
        if let Some(required) = schema["required"].as_array() {
            for field in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(field) {
                    return Err(format!("{path} is missing required argument {field:?}"));
                }
            }
        }
        if schema["additionalProperties"] == false {
            if let Some(unknown) = object.keys().find(|key| !properties.is_some_and(|props| props.contains_key(*key))) {
                return Err(format!("{path} has unknown argument {unknown:?}"));
            }
        }
        for (key, member) in object {
            if let Some(property_schema) = properties.and_then(|props| props.get(key)) {
                validate_schema_value(property_schema, member, &format!("{path}.{key}"))?;
            }
        }
    }
    Ok(())
}

fn list_documents(app: &App) -> Result<Value, String> {
    let mut files = Vec::new();
    walk_documents(&app.root, &app.root, 0, &mut files)?;
    files.sort();
    Ok(json!({"project_root":app.root,"documents":files,"count":files.len()}))
}

fn walk_documents(
    root: &Path,
    dir: &Path,
    depth: usize,
    output: &mut Vec<String>,
) -> Result<(), String> {
    if depth > 16 || output.len() >= 20_000 {
        return Ok(());
    }
    let entries =
        fs::read_dir(dir).map_err(|e| format!("read directory {}: {e}", dir.display()))?;
    for entry in entries {
        if output.len() >= 20_000 {
            break;
        }
        let entry = entry.map_err(|e| format!("read directory entry: {e}"))?;
        let ty = entry
            .file_type()
            .map_err(|e| format!("read file type: {e}"))?;
        if ty.is_symlink() {
            continue;
        }
        let path = entry.path();
        if ty.is_dir() {
            let name = entry.file_name();
            if name == ".git" || name == "target" || name == ".omx" {
                continue;
            }
            walk_documents(root, &path, depth + 1, output)?;
        } else if ty.is_file() && matches!(extension(&path).as_deref(), Some("vxm" | "vxscene")) {
            let canonical =
                fs::canonicalize(&path).map_err(|e| format!("resolve {}: {e}", path.display()))?;
            ensure_under(root, &canonical)?;
            output.push(
                canonical
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

fn inspect_document(app: &App, args: &Value) -> Result<Value, String> {
    let path = existing_path(app, required_string(args, "path")?)?;
    match extension(&path).as_deref() {
        Some("vxm") => {
            let doc: VoxelModelDocument = load_document(&path)?;
            doc.validate()?;
            let mut min = DVec3::splat(f64::INFINITY);
            let mut max = DVec3::splat(f64::NEG_INFINITY);
            for voxel in &doc.voxels {
                min = min.min(voxel.position_m);
                max = max.max(voxel.position_m);
            }
            let materials = doc
                .materials
                .iter()
                .map(|material| &material.name)
                .collect::<Vec<_>>();
            let skeleton = doc.skeleton.as_ref().map(|skeleton| json!({
                "bones": skeleton.bones.iter().enumerate().map(|(index,bone)| json!({"index":index,"name":bone.name,"parent":bone.parent,"rest_translation_m":bone.rest_translation_m,"rest_rotation_xyzw":[bone.rest_rotation.x,bone.rest_rotation.y,bone.rest_rotation.z,bone.rest_rotation.w]})).collect::<Vec<_>>(),
                "voxel_weight_count": skeleton.voxel_weights.len(),
                "clips": skeleton.clips.iter().enumerate().map(|(index,clip)| json!({"index":index,"name":clip.name,"duration_s":clip.duration_s,"tracks":clip.tracks.iter().map(|track| json!({"bone_index":track.bone_index,"keyframes":track.keyframes.iter().map(|key| json!({"time_s":key.time_s,"translation_m":key.translation_m,"rotation_xyzw":[key.rotation.x,key.rotation.y,key.rotation.z,key.rotation.w]})).collect::<Vec<_>>()})).collect::<Vec<_>>()})).collect::<Vec<_>>()
            }));
            Ok(
                json!({"kind":"voxel_model","path":relative(&app.root,&path),"name":doc.name,"valid":true,"voxel_count":doc.voxels.len(),"material_count":doc.materials.len(),"materials":materials,"group_count":doc.groups.len(),"constraint_count":doc.motion_constraints.len(),"cell_size_m":doc.cell_size_m,"bounds_min_m":min,"bounds_max_m":max,"skeleton":skeleton}),
            )
        }
        Some("vxscene") => {
            let doc: SceneDocument = load_document(&path)?;
            doc.validate()?;
            let mut refs = Vec::with_capacity(doc.models.len());
            for instance in &doc.models {
                let asset = referenced_path(
                    app,
                    path.parent().unwrap_or(&app.root),
                    &instance.asset_path,
                )?;
                if extension(&asset).as_deref() != Some("vxm") {
                    return Err(format!(
                        "scene model reference is not a .vxm asset: {}",
                        instance.asset_path.display()
                    ));
                }
                refs.push(relative(&app.root, &asset));
            }
            Ok(
                json!({"kind":"scene","path":relative(&app.root,&path),"name":doc.name,"valid":true,"model_instance_count":doc.models.len(),"model_assets":refs,"scene_config_valid":doc.config.validate().is_ok()}),
            )
        }
        _ => Err("inspect_document supports .vxm and .vxscene files".into()),
    }
}

fn voxelize_mesh(app: &App, args: &Value) -> Result<Value, String> {
    let source = existing_path(app, required_string(args, "source_path")?)?;
    if !matches!(extension(&source).as_deref(), Some("obj" | "glb")) {
        return Err("source_path must refer to an .obj or .glb file".into());
    }
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxm") {
        return Err("output_path must end in .vxm".into());
    }
    let scale = args.get("scale").and_then(Value::as_f64).unwrap_or(1.0);
    if !scale.is_finite() || !(0.01..=100.0).contains(&scale) {
        return Err("scale must be between 0.01 and 100".into());
    }
    let mode = match args.get("mode").and_then(Value::as_str).unwrap_or("shell") {
        "shell" => VoxelizationMode::Shell,
        "filled" => VoxelizationMode::Filled,
        _ => return Err("mode must be shell or filled".into()),
    };
    let name = args
        .get("model_name")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            output
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned)
        })
        .ok_or("output filename is not valid UTF-8")?;
    let mut options = VoxelizationOptions::default();
    options.scale = DVec3::splat(scale);
    options.mode = mode;
    let preview = import_mesh(&source, options)?;
    let material = Material::table()
        .into_iter()
        .find(|m| m.name == "oak")
        .ok_or("built-in oak material profile is unavailable")?;
    let model = preview.to_model_document(name, material)?;
    let voxel_count = model.voxels.len();
    save_document(&output, &model)?;
    Ok(
        json!({"path":relative(&app.root,&output),"valid":true,"voxel_count":voxel_count,"mode":if mode == VoxelizationMode::Shell {"shell"} else {"filled"},"watertight_source":preview.is_watertight,"cell_size_m":preview.cell_size_m,"source_bounds_min_m":preview.source_bounds_min_m,"source_bounds_max_m":preview.source_bounds_max_m}),
    )
}

fn export_model_obj(app: &App, args: &Value) -> Result<Value, String> {
    let source = existing_path(app, required_string(args, "source_path")?)?;
    if extension(&source).as_deref() != Some("vxm") {
        return Err("source_path must refer to a .vxm file".into());
    }
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("obj") {
        return Err("output_path must end in .obj".into());
    }
    let model: VoxelModelDocument = load_document(&source)?;
    model.validate()?;
    let faces = export_obj(&model, &output)?;
    Ok(json!({
        "path":relative(&app.root,&output),
        "source_path":relative(&app.root,&source),
        "valid":true,
        "exposed_face_count":faces,
        "voxel_count":model.voxels.len(),
        "units":"metres"
    }))
}

fn create_model(app: &App, args: &Value) -> Result<Value, String> {
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxm") {
        return Err("output_path must end in .vxm".into());
    }
    let name = args
        .get("model_name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .or_else(|| output.file_stem().and_then(|name| name.to_str()))
        .ok_or("model_name or a UTF-8 output filename is required")?;
    // The document schema requires at least one REV. SceneBuilder is the
    // existing public constructor for a valid material/thermal voxel state.
    let scene = SceneBuilder::standard()
        .add_named_voxel(Vec3::splat(0.005), "oak")?
        .build()?;
    let model = VoxelModelDocument::from_scene(name, &scene);
    model.validate()?;
    let count = model.voxels.len();
    save_document(&output, &model)?;
    Ok(
        json!({"path":relative(&app.root,&output),"valid":true,"name":model.name,"voxel_count":count,"note":"created with one oak REV because empty .vxm documents are invalid"}),
    )
}

fn create_scene(app: &App, args: &Value) -> Result<Value, String> {
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxscene") {
        return Err("output_path must end in .vxscene".into());
    }
    let name = args
        .get("scene_name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .or_else(|| output.file_stem().and_then(|name| name.to_str()))
        .ok_or("scene_name or a UTF-8 output filename is required")?;
    let scene = SceneDocument::new(name, SceneConfig::default())?;
    save_document(&output, &scene)?;
    Ok(
        json!({"path":relative(&app.root,&output),"valid":true,"name":scene.name,"model_instance_count":0}),
    )
}

fn edit_model_skeleton(app: &App, args: &Value) -> Result<Value, String> {
    let source = existing_path(app, required_string(args, "source_path")?)?;
    if extension(&source).as_deref() != Some("vxm") {
        return Err("source_path must refer to a .vxm file".into());
    }
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxm") {
        return Err("output_path must end in .vxm".into());
    }
    let mut model: VoxelModelDocument = load_document(&source)?;
    model.validate()?;
    let operation = required_string(args, "operation")?;
    let mut details = Value::Null;
    match operation {
        "create_root" => {
            if model.skeleton.is_some() {
                return Err("model already has a skeleton".into());
            }
            let name = required_string(args, "name")?;
            let position = array3(args, "position_m")?;
            model.skeleton = Some(SkeletonDocument::with_root_bone(name, position));
        }
        "add_bone" => {
            let skeleton = model.skeleton.as_mut().ok_or("create a root bone first")?;
            let name = required_string(args, "name")?;
            let parent = required_usize(args, "parent_bone")?;
            let position = array3(args, "position_m")?;
            let index = skeleton.add_bone(name, Some(parent), position)?;
            details = json!({"bone_index":index});
        }
        "create_clip" => {
            let skeleton = model
                .skeleton
                .as_mut()
                .ok_or("create at least one bone first")?;
            let name = required_string(args, "name")?;
            let duration = required_f64(args, "duration_s")?;
            let index = skeleton.add_animation_clip(name, duration)?;
            details = json!({"clip_index":index});
        }
        "set_keyframe" => {
            let skeleton = model
                .skeleton
                .as_mut()
                .ok_or("create a skeleton and animation clip first")?;
            let clip = required_usize(args, "clip_index")?;
            let bone = required_usize(args, "bone_index")?;
            let time = required_f64(args, "time_s")?;
            let translation = array3(args, "translation_m")?;
            let rotation = array4(args, "rotation_xyzw")?;
            let rotation = Quat::from_xyzw(
                rotation[0] as f32,
                rotation[1] as f32,
                rotation[2] as f32,
                rotation[3] as f32,
            );
            skeleton.set_keyframe(
                clip,
                bone,
                BoneKeyframeDocument {
                    time_s: time,
                    translation_m: translation,
                    rotation,
                },
            )?;
            details = json!({"clip_index":clip,"bone_index":bone,"time_s":time});
        }
        "auto_weights" => {
            let influences = args
                .get("max_influences")
                .and_then(Value::as_u64)
                .unwrap_or(4) as usize;
            let weights = generate_auto_bone_weights(&model, influences)?;
            let count = weights.len();
            model
                .skeleton
                .as_mut()
                .ok_or("create at least one bone first")?
                .voxel_weights = weights;
            details = json!({"binding_count":count,"max_influences":influences});
        }
        "adjust_weight" => {
            let voxel = required_usize(args, "voxel_index")?;
            let bone = required_usize(args, "bone_index")?;
            let delta = args
                .get("weight_delta")
                .and_then(Value::as_f64)
                .ok_or("weight_delta must be a number")? as f32;
            let skeleton = model.skeleton.as_mut().ok_or("create a skeleton first")?;
            let weight =
                skeleton.adjust_voxel_bone_weight(model.voxels.len(), voxel, bone, delta)?;
            details = json!({"voxel_index":voxel,"bone_index":bone,"resulting_weight":weight});
        }
        _ => return Err(format!("unsupported skeleton operation: {operation}")),
    }
    model.validate()?;
    save_document(&output, &model)?;
    Ok(
        json!({"path":relative(&app.root,&output),"source_path":relative(&app.root,&source),"valid":true,"operation":operation,"details":details,"bone_count":model.skeleton.as_ref().map_or(0,|s|s.bones.len()),"clip_count":model.skeleton.as_ref().map_or(0,|s|s.clips.len())}),
    )
}

/// Builds a complete rig and clip as one atomic MCP operation. The source
/// model is left untouched; only a validated new document is written.
fn author_animation(app: &App, args: &Value) -> Result<Value, String> {
    let source = existing_path(app, required_string(args, "source_path")?)?;
    if extension(&source).as_deref() != Some("vxm") {
        return Err("source_path must refer to a .vxm file".into());
    }
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxm") {
        return Err("output_path must end in .vxm".into());
    }

    let mut model: VoxelModelDocument = load_document(&source)?;
    model.validate()?;
    if model.skeleton.is_some() {
        return Err(
            "source model already has a skeleton; use edit_model_skeleton or a rig-free source"
                .into(),
        );
    }

    let root_name = required_string(args, "root_name")?;
    let root_position = array3(args, "root_position_m")?;
    let bone_specs = args
        .get("bones")
        .and_then(Value::as_array)
        .ok_or("bones must be an array")?;
    if bone_specs.is_empty() {
        return Err("bones must contain at least one child bone".into());
    }

    let mut skeleton = SkeletonDocument::with_root_bone(root_name, root_position);
    for (offset, spec) in bone_specs.iter().enumerate() {
        let name = required_string(spec, "name")?;
        let parent = required_usize(spec, "parent_bone")?;
        // Bones are authored in hierarchy order so every parent already exists.
        if parent >= skeleton.bones.len() {
            return Err(format!(
                "bones[{offset}].parent_bone must refer to the root or an earlier bone"
            ));
        }
        let position = array3(spec, "position_m")?;
        skeleton.add_bone(name, Some(parent), position)?;
    }

    let clip_index = skeleton.add_animation_clip(
        required_string(args, "clip_name")?,
        required_f64(args, "duration_s")?,
    )?;
    let keyframes = args
        .get("keyframes")
        .and_then(Value::as_array)
        .ok_or("keyframes must be an array")?;
    if keyframes.is_empty() {
        return Err("keyframes must contain at least one pose".into());
    }
    for (frame_index, frame) in keyframes.iter().enumerate() {
        let bone_index = required_usize(frame, "bone_index")?;
        let time_s = required_f64(frame, "time_s")?;
        let translation_m = array3(frame, "translation_m")?;
        let rotation = array4(frame, "rotation_xyzw")?;
        let rotation = Quat::from_xyzw(
            rotation[0] as f32,
            rotation[1] as f32,
            rotation[2] as f32,
            rotation[3] as f32,
        );
        skeleton
            .set_keyframe(
                clip_index,
                bone_index,
                BoneKeyframeDocument {
                    time_s,
                    translation_m,
                    rotation,
                },
            )
            .map_err(|error| format!("keyframes[{frame_index}]: {error}"))?;
    }

    let max_influences = args
        .get("max_influences")
        .and_then(Value::as_u64)
        .unwrap_or(4) as usize;
    model.skeleton = Some(skeleton);
    let weights = generate_auto_bone_weights(&model, max_influences)?;
    let weighted_voxels = weights.len();
    let skeleton = model.skeleton.as_mut().expect("skeleton was just assigned");
    skeleton.voxel_weights = weights;
    let bone_count = skeleton.bones.len();
    let clip_name = skeleton.clips[clip_index].name.clone();
    let keyframe_count: usize = skeleton.clips[clip_index]
        .tracks
        .iter()
        .map(|track| track.keyframes.len())
        .sum();
    model.validate()?;
    save_document(&output, &model)?;

    Ok(json!({
        "path":relative(&app.root,&output),
        "source_path":relative(&app.root,&source),
        "valid":true,
        "operation":"author_animation",
        "bone_count":bone_count,
        "clip_count":1,
        "clip_name":clip_name,
        "keyframe_count":keyframe_count,
        "weighted_voxel_count":weighted_voxels,
        "max_influences":max_influences
    }))
}

/// Infer the real body and limb topology and author an IK walk through the engine API.
fn author_procedural_gait(app: &App, args: &Value) -> Result<Value, String> {
    let source = existing_path(app, required_string(args, "source_path")?)?;
    if extension(&source).as_deref() != Some("vxm") {
        return Err("source_path must refer to a .vxm file".into());
    }
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxm") {
        return Err("output_path must end in .vxm".into());
    }
    let up_axis = match args
        .get("up_axis")
        .and_then(Value::as_str)
        .unwrap_or("auto")
    {
        "auto" => None,
        "x" => Some(0),
        "y" => Some(1),
        "z" => Some(2),
        _ => return Err("up_axis must be auto, x, y, or z".into()),
    };
    let mut options = WalkOptions {
        up_axis,
        ..WalkOptions::default()
    };
    if let Some(value) = args.get("clip_name").and_then(Value::as_str) {
        options.clip_name = value.to_owned();
    }
    if args.get("forward_direction").is_some() {
        options.forward_direction = Some(array3(args, "forward_direction")?);
    }
    for (key, target) in [
        ("duration_s", &mut options.duration_s),
        ("stride_m", &mut options.stride_m),
        ("lift_m", &mut options.lift_m),
        ("stance_fraction", &mut options.stance_fraction),
        ("sample_fps", &mut options.sample_fps),
    ] {
        if args.get(key).is_some() {
            *target = required_f64(args, key)?;
        }
    }
    let mut model: VoxelModelDocument = load_document(&source)?;
    let report = author_walk(&mut model, &options)?;
    model.name = format!(
        "{} | {}",
        source.file_stem().unwrap_or_default().to_string_lossy(),
        options.clip_name
    );
    model.validate()?;
    save_document(&output, &model)?;
    Ok(json!({
        "path":relative(&app.root,&output), "source_path":relative(&app.root,&source),
        "valid":true, "operation":"author_procedural_gait", "motion":"in_place_ik_walk",
        "clip_name":options.clip_name, "max_influences":1, "report":report
    }))
}

fn sample_animation(app: &App, args: &Value) -> Result<Value, String> {
    let path = existing_path(app, required_string(args, "path")?)?;
    if extension(&path).as_deref() != Some("vxm") {
        return Err("path must refer to a .vxm file".into());
    }
    let model: VoxelModelDocument = load_document(&path)?;
    model.validate()?;
    let skeleton = model.skeleton.as_ref().ok_or("model has no skeleton")?;
    let clip_index = required_usize(args, "clip_index")?;
    let time_s = required_f64(args, "time_s")?;
    let clip = skeleton
        .clips
        .get(clip_index)
        .ok_or("animation clip index is out of range")?;
    let material_limits = model
        .skeleton_material_limits()
        .ok_or("model has no skeleton material profile")?;
    let sampled_poses =
        skeleton.bone_local_poses_at_with_material_limits(clip_index, time_s, &material_limits)?;
    let poses = skeleton.bones.iter().zip(&sampled_poses).enumerate().map(|(bone_index, (bone, pose))| {
        Ok(json!({"bone_index":bone_index,"name":bone.name,"parent":bone.parent,"translation_m":pose.translation_m,"rotation_xyzw":[pose.rotation.x,pose.rotation.y,pose.rotation.z,pose.rotation.w]}))
    }).collect::<Result<Vec<_>, String>>()?;
    let global_positions = skeleton.bone_global_positions_at_with_material_limits(
        clip_index,
        time_s,
        Some(&material_limits),
    )?;
    Ok(
        json!({"path":relative(&app.root,&path),"clip_index":clip_index,"clip_name":clip.name,"requested_time_s":time_s,"sampled_time_s":time_s.clamp(0.0,clip.duration_s),"duration_s":clip.duration_s,"bones":poses,"global_joint_positions_m":global_positions,"material_limits":material_limits}),
    )
}

fn edit_voxel_model(app: &App, args: &Value) -> Result<Value, String> {
    let source = existing_path(app, required_string(args, "source_path")?)?;
    if extension(&source).as_deref() != Some("vxm") {
        return Err("source_path must refer to a .vxm file".into());
    }
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxm") {
        return Err("output_path must end in .vxm".into());
    }
    let mut model: VoxelModelDocument = load_document(&source)?;
    model.validate()?;
    let operation = required_string(args, "operation")?;
    let result;
    match operation {
        "add_voxel" => {
            let cell = int_array3(args, "cell")?;
            let material_name = args
                .get("material_name")
                .and_then(Value::as_str)
                .unwrap_or("oak");
            let index = add_model_voxel(&mut model, cell, material_name)?;
            result = json!({"voxel_index":index,"cell":cell,"material":material_name});
        }
        "remove_voxel" => {
            let index = required_usize(args, "voxel_index")?;
            remove_model_voxel(&mut model, index)?;
            result = json!({"removed_voxel_index":index});
        }
        "move_voxel" => {
            let index = required_usize(args, "voxel_index")?;
            let cell = int_array3(args, "cell")?;
            if model.motion_constraints.iter().any(|constraint| {
                constraint.a == ConstraintPartDocument::Voxel(index as u32)
                    || constraint.b == ConstraintPartDocument::Voxel(index as u32)
            }) {
                return Err("cannot move a constrained REV independently; move its constrained group instead".into());
            }
            let target = cell_center(cell, model.cell_size_m);
            ensure_unoccupied(&model, target, Some(index))?;
            let voxel = model
                .voxels
                .get_mut(index)
                .ok_or("voxel_index is out of range")?;
            let delta = target - voxel.position_m;
            voxel.position_m = target;
            voxel.rest_position_m += delta;
            result = json!({"voxel_index":index,"cell":cell,"position_m":target});
        }
        "duplicate_voxel" => {
            let source_index = required_usize(args, "voxel_index")?;
            let cell = int_array3(args, "cell")?;
            let mut duplicate = model
                .voxels
                .get(source_index)
                .ok_or("voxel_index is out of range")?
                .clone();
            let target = cell_center(cell, model.cell_size_m);
            ensure_unoccupied(&model, target, None)?;
            let delta = target - duplicate.position_m;
            duplicate.position_m = target;
            duplicate.rest_position_m += delta;
            let index = model.voxels.len();
            model.voxels.push(duplicate);
            for group in &mut model.groups {
                if group.voxels.contains(&(source_index as u32)) {
                    group.voxels.push(index as u32);
                }
            }
            if let Some(skeleton) = model.skeleton.as_mut() {
                let copied = skeleton
                    .voxel_weights
                    .iter()
                    .filter(|binding| binding.voxel_index as usize == source_index)
                    .map(
                        |binding| voxel_physics_lab::documents::VoxelBoneWeightDocument {
                            voxel_index: index as u32,
                            bone_index: binding.bone_index,
                            weight: binding.weight,
                        },
                    )
                    .collect::<Vec<_>>();
                skeleton.voxel_weights.extend(copied);
            }
            result = json!({"source_voxel_index":source_index,"voxel_index":index,"cell":cell});
        }
        "assign_material" => {
            let index = required_usize(args, "voxel_index")?;
            let name = required_string(args, "material_name")?;
            let material_index = model
                .materials
                .iter()
                .position(|material| material.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| {
                    format!("material {name:?} is not present in this model's material table")
                })?;
            let voxel = model
                .voxels
                .get_mut(index)
                .ok_or("voxel_index is out of range")?;
            voxel.state.composition = Composition::pure(
                u16::try_from(material_index)
                    .map_err(|_| "material index exceeds supported range")?,
            );
            result = json!({"voxel_index":index,"material":model.materials[material_index].name,"mass_kg_preserved":voxel.state.mass,"internal_energy_j_preserved":voxel.state.total_internal_energy_j()});
        }
        "add_material" => {
            let name = required_string(args, "material_name")?;
            let (material_index, already_present) = if let Some(index) = model
                .materials
                .iter()
                .position(|material| material.name.eq_ignore_ascii_case(name))
            {
                (index, true)
            } else {
                let preset = Material::table()
                    .into_iter()
                    .find(|material| material.name.eq_ignore_ascii_case(name))
                    .ok_or_else(|| format!("unknown built-in material {name:?}"))?;
                if model.materials.len() > u16::MAX as usize {
                    return Err(
                        "model already contains the maximum supported material count".into(),
                    );
                }
                model.materials.push(preset);
                (model.materials.len() - 1, false)
            };
            result = json!({"material_index":material_index,"material":model.materials[material_index].name,"already_present":already_present});
        }
        "stamp_shape" => {
            let shape = required_string(args, "shape")?;
            let min_cell = int_array3(args, "min_cell")?;
            let max_cell = int_array3(args, "max_cell")?;
            let material_name = args
                .get("material_name")
                .and_then(Value::as_str)
                .unwrap_or("oak");
            let cells = shape_cells(shape, min_cell, max_cell)?;
            let mut occupied = HashSet::with_capacity(model.voxels.len() + cells.len());
            for voxel in &model.voxels {
                occupied.insert(model_cell(voxel.position_m, model.cell_size_m));
            }
            for cell in &cells {
                if !occupied.insert(*cell) {
                    return Err(format!(
                        "shape overlaps an occupied cell at {cell:?}; no output was written"
                    ));
                }
            }
            let material_index = model
                .materials
                .iter()
                .position(|material| material.name.eq_ignore_ascii_case(material_name))
                .ok_or_else(|| {
                    format!(
                        "material {material_name:?} is not present in this model's material table"
                    )
                })?;
            let template = make_voxel_template(&model, material_index)?;
            let first_index = model.voxels.len();
            for cell in &cells {
                let mut voxel = template.clone();
                voxel.position_m = cell_center(*cell, model.cell_size_m);
                voxel.rest_position_m = voxel.position_m;
                model.voxels.push(voxel);
            }
            result = json!({"shape":shape,"added_voxel_count":cells.len(),"first_voxel_index":first_index,"min_cell":min_cell,"max_cell":max_cell,"material":model.materials[material_index].name});
        }
        _ => return Err(format!("unsupported voxel-model operation: {operation}")),
    }
    model.validate()?;
    let resulting = voxel_model_summary(&model);
    save_document(&output, &model)?;
    Ok(
        json!({"path":relative(&app.root,&output),"source_path":relative(&app.root,&source),"valid":true,"operation":operation,"result":result,"state":resulting}),
    )
}

fn add_model_voxel(
    model: &mut VoxelModelDocument,
    cell: [i32; 3],
    material_name: &str,
) -> Result<usize, String> {
    let target = cell_center(cell, model.cell_size_m);
    ensure_unoccupied(model, target, None)?;
    let material_index = model
        .materials
        .iter()
        .position(|entry| entry.name.eq_ignore_ascii_case(material_name))
        .ok_or_else(|| {
            format!("material {material_name:?} is not present in this model's material table")
        })?;
    let mut voxel = make_voxel_template(model, material_index)?;
    voxel.position_m = target;
    voxel.rest_position_m = target;
    let index = model.voxels.len();
    model.voxels.push(voxel);
    Ok(index)
}

fn make_voxel_template(
    model: &VoxelModelDocument,
    material_index: usize,
) -> Result<ModelVoxel, String> {
    let material = model
        .materials
        .get(material_index)
        .ok_or("material index is out of range")?;
    let template_scene = SceneBuilder::standard()
        .add_named_voxel(Vec3::splat(0.005), "oak")?
        .build()?;
    let template_model = VoxelModelDocument::from_scene("voxel-template", &template_scene);
    let mut voxel = template_model
        .voxels
        .into_iter()
        .next()
        .ok_or("voxel template builder returned no REV")?;
    voxel.state.composition = Composition::pure(
        u16::try_from(material_index).map_err(|_| "material index exceeds supported range")?,
    );
    voxel.state.mass = (material.density as f64 * model.cell_size_m.powi(3)) as f32;
    voxel.state.internal_energy = material.energy_at_temperature(voxel.state.mass, 293.15);
    voxel.state.internal_energy_remainder_j = 0.0;
    Ok(voxel)
}

fn shape_cells(shape: &str, min: [i32; 3], max: [i32; 3]) -> Result<Vec<[i32; 3]>, String> {
    let mut bounds_min = [0_i64; 3];
    let mut bounds_max = [0_i64; 3];
    let mut volume = 1_u64;
    for axis in 0..3 {
        if min[axis] > max[axis] {
            return Err("min_cell coordinates must be <= max_cell coordinates".into());
        }
        bounds_min[axis] = i64::from(min[axis]);
        bounds_max[axis] = i64::from(max[axis]);
        volume = volume
            .checked_mul((bounds_max[axis] - bounds_min[axis] + 1) as u64)
            .ok_or("shape bounds are too large")?;
    }
    if volume > 100_000 {
        return Err("shape bounds exceed the 100,000-cell per-operation limit".into());
    }
    if !matches!(shape, "block" | "sphere" | "cylinder" | "frame") {
        return Err("shape must be block, sphere, cylinder, or frame".into());
    }
    let center =
        std::array::from_fn::<_, 3, _>(|axis| (bounds_min[axis] + bounds_max[axis]) as f64 * 0.5);
    let radius = std::array::from_fn::<_, 3, _>(|axis| {
        (bounds_max[axis] - bounds_min[axis] + 1) as f64 * 0.5
    });
    let mut cells = Vec::new();
    for z in bounds_min[2]..=bounds_max[2] {
        for y in bounds_min[1]..=bounds_max[1] {
            for x in bounds_min[0]..=bounds_max[0] {
                let edge = x == bounds_min[0]
                    || x == bounds_max[0]
                    || y == bounds_min[1]
                    || y == bounds_max[1]
                    || z == bounds_min[2]
                    || z == bounds_max[2];
                let nx = (x as f64 - center[0]) / radius[0];
                let ny = (y as f64 - center[1]) / radius[1];
                let nz = (z as f64 - center[2]) / radius[2];
                let include = match shape {
                    "block" => true,
                    "frame" => edge,
                    "sphere" => nx * nx + ny * ny + nz * nz <= 1.0,
                    "cylinder" => nx * nx + nz * nz <= 1.0,
                    _ => false,
                };
                if include {
                    cells.push([x as i32, y as i32, z as i32]);
                }
            }
        }
    }
    if cells.is_empty() {
        return Err("shape bounds produced no voxels".into());
    }
    Ok(cells)
}

fn model_cell(position_m: DVec3, cell_size_m: f64) -> [i32; 3] {
    ((position_m / cell_size_m - DVec3::splat(0.5)).round())
        .as_ivec3()
        .to_array()
}

fn remove_model_voxel(model: &mut VoxelModelDocument, index: usize) -> Result<(), String> {
    if index >= model.voxels.len() {
        return Err("voxel_index is out of range".into());
    }
    if model.voxels.len() == 1 {
        return Err(
            "cannot remove the final REV: .vxm documents must contain at least one voxel".into(),
        );
    }
    for group in &model.groups {
        if group.voxels.contains(&(index as u32)) && group.voxels.len() == 1 {
            return Err(format!(
                "cannot remove voxel {index}: it is the only member of group {:?}",
                group.name
            ));
        }
    }
    for constraint in &model.motion_constraints {
        if constraint.a == ConstraintPartDocument::Voxel(index as u32)
            || constraint.b == ConstraintPartDocument::Voxel(index as u32)
        {
            return Err(format!(
                "cannot remove voxel {index}: a motion constraint references it"
            ));
        }
    }
    model.voxels.remove(index);
    for group in &mut model.groups {
        group.voxels.retain(|&voxel| voxel as usize != index);
        for voxel in &mut group.voxels {
            if *voxel as usize > index {
                *voxel -= 1;
            }
        }
    }
    for constraint in &mut model.motion_constraints {
        for part in [&mut constraint.a, &mut constraint.b] {
            if let ConstraintPartDocument::Voxel(voxel) = part {
                if *voxel as usize > index {
                    *voxel -= 1;
                }
            }
        }
    }
    if let Some(skeleton) = model.skeleton.as_mut() {
        skeleton
            .voxel_weights
            .retain(|binding| binding.voxel_index as usize != index);
        for binding in &mut skeleton.voxel_weights {
            if binding.voxel_index as usize > index {
                binding.voxel_index -= 1;
            }
        }
    }
    Ok(())
}

fn ensure_unoccupied(
    model: &VoxelModelDocument,
    target: DVec3,
    except: Option<usize>,
) -> Result<(), String> {
    if model.voxels.iter().enumerate().any(|(index, voxel)| {
        Some(index) != except && voxel.position_m.distance_squared(target) < 1e-18
    }) {
        return Err("target REV cell is already occupied".into());
    }
    Ok(())
}

fn cell_center(cell: [i32; 3], cell_size_m: f64) -> DVec3 {
    DVec3::new(cell[0] as f64, cell[1] as f64, cell[2] as f64) * cell_size_m
        + DVec3::splat(cell_size_m * 0.5)
}

fn voxel_model_summary(model: &VoxelModelDocument) -> Value {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for voxel in &model.voxels {
        min = min.min(voxel.position_m);
        max = max.max(voxel.position_m);
    }
    json!({"name":model.name,"voxel_count":model.voxels.len(),"material_count":model.materials.len(),"group_count":model.groups.len(),"constraint_count":model.motion_constraints.len(),"bounds_min_m":min,"bounds_max_m":max})
}

fn edit_scene(app: &App, args: &Value) -> Result<Value, String> {
    let source = existing_path(app, required_string(args, "source_path")?)?;
    if extension(&source).as_deref() != Some("vxscene") {
        return Err("source_path must refer to a .vxscene file".into());
    }
    let output = output_path(app, required_string(args, "output_path")?)?;
    if extension(&output).as_deref() != Some("vxscene") {
        return Err("output_path must end in .vxscene".into());
    }
    let mut scene: SceneDocument = load_document(&source)?;
    scene.validate()?;
    // Scene paths are relative to their document directory; remap references
    // when the new version is written elsewhere.
    let source_dir = source.parent().unwrap_or(&app.root);
    let output_dir = output
        .parent()
        .ok_or("scene output path has no parent directory")?;
    let mut instance_voxel_counts = Vec::new();
    for instance in &mut scene.models {
        let asset = referenced_path(app, source_dir, &instance.asset_path)?;
        if extension(&asset).as_deref() != Some("vxm") {
            return Err(format!(
                "scene reference is not a .vxm asset: {}",
                instance.asset_path.display()
            ));
        }
        let model: VoxelModelDocument = load_document(&asset)?;
        model.validate()?;
        instance_voxel_counts.push(model.voxels.len());
        instance.asset_path = relative_asset_path(output_dir, &asset)?;
    }
    let operation = required_string(args, "operation")?;
    let result;
    match operation {
        "add_instance" => {
            let asset = existing_path(app, required_string(args, "asset_path")?)?;
            if extension(&asset).as_deref() != Some("vxm") {
                return Err("asset_path must refer to a .vxm file".into());
            }
            let asset_doc: VoxelModelDocument = load_document(&asset)?;
            asset_doc.validate()?;
            let stored_asset_path = relative_asset_path(output_dir, &asset)?;
            let translation = if args.get("translation_m").is_some() {
                array3(args, "translation_m")?
            } else {
                DVec3::ZERO
            };
            let rotation = if args.get("rotation_xyzw").is_some() {
                let q = array4(args, "rotation_xyzw")?;
                Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32)
            } else {
                Quat::IDENTITY
            };
            let scale = if args.get("scale").is_some() {
                array3_f32(args, "scale")?
            } else {
                Vec3::ONE
            };
            let index = scene.models.len();
            scene.add_model(ModelInstanceDocument {
                asset_path: stored_asset_path,
                translation_m: translation,
                rotation,
                scale,
            })?;
            result = json!({"instance_index":index,"asset_path":relative(&app.root,&asset)});
        }
        "move_instance" => {
            let index = required_usize(args, "instance_index")?;
            let delta = int_array3(args, "delta_cells")?;
            scene.translate_model_cells(index, glam::IVec3::from_array(delta))?;
            result = json!({"instance_index":index,"delta_cells":delta});
        }
        "duplicate_instance" => {
            let index = required_usize(args, "instance_index")?;
            let delta = int_array3(args, "delta_cells")?;
            let new_index = scene.duplicate_model_cells(index, glam::IVec3::from_array(delta))?;
            result = json!({"source_instance_index":index,"instance_index":new_index,"delta_cells":delta});
        }
        "remove_instance" => {
            let index = required_usize(args, "instance_index")?;
            let removed = scene.remove_model(index)?;
            result = json!({"removed_instance_index":index,"asset_path":removed.asset_path});
        }
        "add_manual_bond" | "set_manual_bond" => {
            let rest_length_m = if args.get("rest_length_m").is_some() {
                required_f64(args, "rest_length_m")?
            } else {
                0.01
            };
            let pretension_n = if args.get("pretension_n").is_some() {
                required_f64(args, "pretension_n")?
            } else {
                0.0
            };
            if operation == "add_manual_bond" {
                let a = ManualBondEndpoint {
                    instance: required_usize(args, "instance_a")?,
                    voxel: u32::try_from(required_usize(args, "voxel_a")?)
                        .map_err(|_| "voxel_a is too large")?,
                };
                let b = ManualBondEndpoint {
                    instance: required_usize(args, "instance_b")?,
                    voxel: u32::try_from(required_usize(args, "voxel_b")?)
                        .map_err(|_| "voxel_b is too large")?,
                };
                for end in [a, b] {
                    let count = instance_voxel_counts
                        .get(end.instance)
                        .ok_or("manual bond refers to a missing model instance")?;
                    if end.voxel as usize >= *count {
                        return Err(format!(
                            "REV {} is out of range for instance {} ({count} REVs)",
                            end.voxel, end.instance
                        ));
                    }
                }
                let mut bond = ManualBondDocument::new(a, b);
                bond.rest_length_m = rest_length_m;
                bond.pretension_n = pretension_n;
                let index = scene.add_manual_bond(bond)?;
                result = json!({"bond_index":index,"rest_length_m":rest_length_m,"pretension_n":pretension_n,"note":"representability (one bond per axis per REV) is checked when the scene is built"});
            } else {
                let index = required_usize(args, "bond_index")?;
                scene.set_manual_bond_state(index, rest_length_m, pretension_n)?;
                result = json!({"bond_index":index,"rest_length_m":rest_length_m,"pretension_n":pretension_n});
            }
        }
        "remove_manual_bond" => {
            let index = required_usize(args, "bond_index")?;
            scene.remove_manual_bond(index)?;
            result = json!({"removed_bond_index":index});
        }
        "make_unique" => {
            let index = required_usize(args, "instance_index")?;
            let target = output_path(app, required_string(args, "new_asset_path")?)?;
            if extension(&target).as_deref() != Some("vxm") {
                return Err("new_asset_path must end in .vxm".into());
            }
            let stored = relative_asset_path(output_dir, &target)?;
            // The scene output is written only after the copy succeeds; the
            // copy itself is exclusive, so an existing file is never replaced.
            let created = scene.make_unique(index, &output, &stored)?;
            result = json!({"instance_index":index,"new_asset_path":relative(&app.root,&created),"note":"only this instance now uses the copy; other instances keep the original"});
        }
        _ => return Err(format!("unsupported scene operation: {operation}")),
    }
    scene.validate()?;
    let mut state = Vec::with_capacity(scene.models.len());
    for (index, instance) in scene.models.iter().enumerate() {
        let asset = referenced_path(app, output_dir, &instance.asset_path)?;
        state.push(json!({"instance_index":index,"asset_path":relative(&app.root,&asset),"translation_m":instance.translation_m,"rotation_xyzw":[instance.rotation.x,instance.rotation.y,instance.rotation.z,instance.rotation.w],"scale":[instance.scale.x,instance.scale.y,instance.scale.z]}));
    }
    save_document(&output, &scene)?;
    Ok(
        json!({"path":relative(&app.root,&output),"source_path":relative(&app.root,&source),"valid":true,"operation":operation,"result":result,"state":{"name":scene.name,"model_instance_count":scene.models.len(),"models":state}}),
    )
}

fn relative_asset_path(from_dir: &Path, asset: &Path) -> Result<PathBuf, String> {
    let from = from_dir.components().collect::<Vec<_>>();
    let to = asset.components().collect::<Vec<_>>();
    if from.first() != to.first() {
        return Err("scene and model asset are on different path roots".into());
    }
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let mut relative = PathBuf::new();
    for _ in common..from.len() {
        relative.push("..");
    }
    for component in &to[common..] {
        relative.push(component.as_os_str());
    }
    Ok(relative)
}

fn array3(args: &Value, key: &str) -> Result<DVec3, String> {
    let values = args
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array of three numbers"))?;
    if values.len() != 3 {
        return Err(format!("{key} must contain exactly three numbers"));
    }
    let xyz = values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| format!("{key} values must be numbers"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DVec3::new(xyz[0], xyz[1], xyz[2]))
}

fn array4(args: &Value, key: &str) -> Result<[f64; 4], String> {
    let values = args
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array of four numbers"))?;
    if values.len() != 4 {
        return Err(format!("{key} must contain exactly four numbers"));
    }
    let mut result = [0.0; 4];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_f64()
            .ok_or_else(|| format!("{key} values must be numbers"))?;
    }
    Ok(result)
}

fn array3_f32(args: &Value, key: &str) -> Result<Vec3, String> {
    let value = array3(args, key)?;
    Ok(Vec3::new(value.x as f32, value.y as f32, value.z as f32))
}

fn int_array3(args: &Value, key: &str) -> Result<[i32; 3], String> {
    let values = args
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array of three integers"))?;
    if values.len() != 3 {
        return Err(format!("{key} must contain exactly three integers"));
    }
    let mut result = [0_i32; 3];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| format!("{key} values must fit signed 32-bit integers"))?;
    }
    Ok(result)
}

fn required_usize(args: &Value, key: &str) -> Result<usize, String> {
    args.get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{key} must be a non-negative integer"))
}

fn required_f64(args: &Value, key: &str) -> Result<f64, String> {
    args.get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{key} must be a finite number"))
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn existing_path(app: &App, path: impl AsRef<Path>) -> Result<PathBuf, String> {
    let input = path.as_ref();
    if input.is_absolute() {
        return Err("absolute paths are not accepted; use a path beneath the project root".into());
    }
    let canonical = fs::canonicalize(app.root.join(input))
        .map_err(|e| format!("resolve {}: {e}", input.display()))?;
    ensure_under(&app.root, &canonical)?;
    if !canonical.is_file() {
        return Err(format!("not a file: {}", input.display()));
    }
    Ok(canonical)
}

fn referenced_path(app: &App, base: &Path, path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Err("absolute asset references are not accepted".into());
    }
    let canonical = fs::canonicalize(base.join(path))
        .map_err(|error| format!("resolve referenced asset {}: {error}", path.display()))?;
    ensure_under(&app.root, &canonical)?;
    if !canonical.is_file() {
        return Err(format!(
            "referenced asset is not a file: {}",
            path.display()
        ));
    }
    Ok(canonical)
}

fn output_path(app: &App, input: &str) -> Result<PathBuf, String> {
    let path = Path::new(input);
    if path.is_absolute() {
        return Err("absolute output paths are not accepted".into());
    }
    if path.components().any(|c| {
        matches!(
            c,
            std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_)
        )
    }) {
        return Err("output_path cannot escape the project root".into());
    }
    let joined = app.root.join(path);
    if joined.exists() {
        return Err("output_path already exists; refusing to overwrite a project file".into());
    }
    let parent = joined
        .parent()
        .ok_or("output_path has no parent directory")?;
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|e| format!("output directory must already exist: {e}"))?;
    ensure_under(&app.root, &canonical_parent)?;
    Ok(canonical_parent.join(joined.file_name().ok_or("output_path has no filename")?))
}

fn ensure_under(root: &Path, path: &Path) -> Result<(), String> {
    if !path.starts_with(root) {
        Err("path resolves outside the project root".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod runtime_mcp_tests {
    use super::*;

    fn app(enabled: bool) -> App {
        let root = PathBuf::from(format!(
            "E:/runtime-mcp-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        fs::write(
            root.join(SETTINGS_FILE),
            json!({"version":1,"mcp_enabled":enabled}).to_string(),
        )
        .unwrap();
        App { root }
    }

    #[test]
    fn runtime_tools_are_listed_as_read_only_and_have_strict_schemas() {
        let tools = tool_definitions();
        for name in ["get_runtime_status", "read_runtime_metrics", "get_runtime_objects"] {
            let definition = tools
                .iter()
                .find(|tool| tool["name"] == name)
                .expect("runtime tool definition");
            assert_eq!(definition["annotations"]["readOnlyHint"], true);
            assert_eq!(definition["inputSchema"]["additionalProperties"], false);
        }

        assert!(validate_tool_arguments("get_runtime_status", &json!({})).is_ok());
        assert!(validate_tool_arguments("read_runtime_metrics", &json!({})).is_ok());
        assert!(validate_tool_arguments(
            "read_runtime_metrics",
            &json!({"after_sequence":0,"max_samples":64,"wait_ms":10000})
        )
        .is_ok());
        assert!(validate_tool_arguments(
            "read_runtime_metrics",
            &json!({"max_samples":65})
        )
        .unwrap_err()
        .contains("above its maximum"));
        assert!(validate_tool_arguments(
            "read_runtime_metrics",
            &json!({"unexpected":true})
        )
        .unwrap_err()
        .contains("unknown argument"));
        assert!(validate_tool_arguments(
            "read_runtime_metrics",
            &json!({"topic":"event"})
        ).is_ok());
    }

    #[test]
    fn batch_actions_chain_create_and_inspect() {
        let app = app(true);
        let response = call_tool(&app, &json!({"name":"batch_actions","arguments":{"actions":[
            {"id":"create","name":"create_model","arguments":{"output_path":"batch.vxm"}},
            {"id":"inspect","name":"inspect_document","arguments":{"path":"batch.vxm"}}
        ]}})).unwrap();
        assert_eq!(response["isError"], false, "batch response: {response}");
        let content = &response["structuredContent"];
        assert_eq!(content["action_count"], 2);
        assert_eq!(content["success_count"], 2, "batch response: {response}");
        assert_eq!(content["results"][1]["result"]["valid"], true, "batch response: {response}");
        let _ = fs::remove_dir_all(app.root);
    }

    #[test]
    fn batch_prevalidates_every_action_before_writing() {
        let app = app(true);
        let response = call_tool(&app, &json!({"name":"batch_actions","arguments":{"actions":[
            {"name":"create_model","arguments":{"output_path":"must-not-exist.vxm"}},
            {"name":"create_scene","arguments":{}}
        ]}})).unwrap();
        assert_eq!(response["isError"], true);
        assert!(!app.root.join("must-not-exist.vxm").exists());
        let _ = fs::remove_dir_all(app.root);
    }

    #[test]
    fn batch_stops_or_continues_after_runtime_action_errors() {
        let app = app(true);
        let actions = json!([
            {"name":"inspect_document","arguments":{"path":"missing.vxm"}},
            {"name":"create_model","arguments":{"output_path":"after-error.vxm"}}
        ]);
        let stopped = call_tool(&app, &json!({"name":"batch_actions","arguments":{"actions":actions.clone()}})).unwrap();
        assert_eq!(stopped["structuredContent"]["results"][1]["status"], "skipped", "batch response: {stopped}");
        assert!(!app.root.join("after-error.vxm").exists());

        let continued = call_tool(&app, &json!({"name":"batch_actions","arguments":{"stop_on_error":false,"actions":actions}})).unwrap();
        assert_eq!(continued["structuredContent"]["error_count"], 1, "batch response: {continued}");
        assert_eq!(continued["structuredContent"]["success_count"], 1, "batch response: {continued}");
        assert!(app.root.join("after-error.vxm").is_file());
        let _ = fs::remove_dir_all(app.root);
    }

    #[test]
    fn batch_response_budget_marks_omitted_results_without_losing_receipts() {
        let mut results = vec![json!({"index":0,"id":"large","name":"list_documents","status":"success","result":{"documents":"x".repeat(1_100_000),"path":"project"}})];
        compact_batch_results(&mut results);
        assert_eq!(results[0]["status"], "success");
        assert_eq!(results[0]["result_omitted"], true);
        assert_eq!(results[0]["result"]["path"], "project");
        assert!(tool_result(json!({"results":results})).to_string().len() <= 1_000_000);
    }

    #[test]
    fn batch_rejects_nested_unknown_disabled_and_excess_wait_before_execution() {
        let app = app(true);
        for (actions, message) in [
            (json!([{"name":"batch_actions","arguments":{"actions":[]}}]), "nested"),
            (json!([{"name":"no_such_tool","arguments":{}}]), "unknown tool"),
            (json!([{"name":"read_runtime_metrics","arguments":{"wait_ms":10000}},{"name":"read_runtime_metrics","arguments":{"wait_ms":1}}]), "combined"),
        ] {
            let mut with_write = vec![json!({"name":"create_model","arguments":{"output_path":"must-not-exist.vxm"}})];
            with_write.extend(actions.as_array().unwrap().iter().cloned());
            let response = call_tool(&app, &json!({"name":"batch_actions","arguments":{"actions":with_write}})).unwrap();
            assert_eq!(response["isError"], true);
            assert!(response["content"][0]["text"].as_str().unwrap().contains(message));
            assert!(!app.root.join("must-not-exist.vxm").exists());
        }
        fs::write(app.root.join(SETTINGS_FILE), json!({"version":1,"mcp_enabled":false}).to_string()).unwrap();
        let disabled = call_tool(&app, &json!({"name":"batch_actions","arguments":{"actions":[
            {"name":"create_model","arguments":{"output_path":"must-not-exist.vxm"}}
        ]}})).unwrap();
        assert_eq!(disabled["isError"], true);
        assert!(disabled["content"][0]["text"].as_str().unwrap().contains("disabled"));
        assert!(!app.root.join("must-not-exist.vxm").exists());
        let _ = fs::remove_dir_all(app.root);
    }

    #[test]
    fn runtime_tools_route_when_enabled_and_fail_closed_when_disabled() {
        let enabled = app(true);
        let listed = dispatch_method(&enabled, "tools/list", &json!({}), false).unwrap();
        assert!(listed["tools"].as_array().unwrap().iter().any(|tool| {
            tool["name"] == "read_runtime_metrics"
        }));

        fs::write(enabled.root.join(SETTINGS_FILE), json!({"version":1,"mcp_enabled":false}).to_string()).unwrap();
        let listed = dispatch_method(&enabled, "tools/list", &json!({}), false).unwrap();
        assert_eq!(listed["tools"], json!([]));
        let result = call_tool(
            &enabled,
            &json!({"name":"read_runtime_metrics","arguments":{}}),
        )
        .unwrap();
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("disabled"));

        fs::write(enabled.root.join(SETTINGS_FILE), json!({"version":1,"mcp_enabled":true}).to_string()).unwrap();
        let listed = dispatch_method(&enabled, "tools/list", &json!({}), false).unwrap();
        assert!(listed["tools"].as_array().unwrap().iter().any(|tool| tool["name"] == "read_runtime_metrics"));
        let _ = fs::remove_dir_all(enabled.root);
    }

    #[test]
    fn runtime_tools_query_a_mock_loopback_runtime() {
        let root = PathBuf::from(format!(
            "E:/voxel-editor-mcp-runtime-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let server = voxel_physics_lab::runtime_metrics::start_server(&root).unwrap();
        let initial = voxel_physics_lab::runtime_metrics::query_runtime(&root, &json!({"command":"status"})).unwrap();
        let cursor = initial["latest_sequence"].as_u64().unwrap();
        let session = initial["session_id"].clone();
        voxel_physics_lab::runtime_metrics::publish("physics", json!({"step_count":7,"objects":[{"id":2}]}));
        let app = App { root: root.clone() };

        let status = call_tool(&app, &json!({"name":"get_runtime_status","arguments":{}})).unwrap();
        assert_eq!(status["structuredContent"]["session_id"], session);
        let metrics = call_tool(&app, &json!({"name":"read_runtime_metrics","arguments":{"after_sequence":cursor,"max_samples":8,"topic":"physics"}})).unwrap();
        assert_eq!(metrics["structuredContent"]["samples"][0]["data"]["step_count"], 7);
        let objects = call_tool(&app, &json!({"name":"get_runtime_objects","arguments":{}})).unwrap();
        assert_eq!(objects["structuredContent"]["objects"][0]["id"], 2);

        drop(server);
        let _ = fs::remove_dir_all(root);
    }
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
}
fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
fn tool_result(value: Value) -> Value {
    json!({"content":[{"type":"text","text":value.to_string()}],"structuredContent":value,"isError":false})
}
fn tool_error(message: &str) -> Value {
    json!({"content":[{"type":"text","text":message}],"isError":true})
}
fn write_message(writer: &mut impl Write, value: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, value).map_err(|e| format!("encode JSON-RPC: {e}"))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|e| format!("write JSON-RPC: {e}"))
}

#[cfg(test)]
mod editor_controls_tests {
    use super::*;
    use voxel_physics_lab::{
        creature::{genome_for_body, OrganParams, SensorLayout},
        creature_inspector::{CreatureSpec, CREATURE_FILE_VERSION},
        documents::VoxelBoneWeightDocument,
        run_panel::{RunPanel, RunPanelInput, RunState, SolverMode},
        scene::VoxelSpawn,
    };

    fn app() -> App {
        let root = PathBuf::from(format!(
            "E:/controls-mcp-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        fs::write(
            root.join(SETTINGS_FILE),
            json!({"version":1,"mcp_enabled":true}).to_string(),
        )
        .unwrap();
        App { root }
    }

    fn call(app: &App, name: &str, arguments: Value) -> Value {
        call_tool(app, &json!({"name":name,"arguments":arguments})).unwrap()
    }

    fn ok(app: &App, name: &str, arguments: Value) -> Value {
        let result = call(app, name, arguments);
        assert_ne!(result["isError"], true, "{name} failed: {}", result["content"][0]["text"]);
        result["structuredContent"].clone()
    }

    fn error_text(app: &App, name: &str, arguments: Value) -> String {
        let result = call(app, name, arguments);
        assert_eq!(result["isError"], true, "{name} unexpectedly succeeded");
        result["content"][0]["text"].as_str().unwrap().to_owned()
    }

    fn new_scene(app: &App, path: &str) {
        ok(app, "create_scene", json!({"output_path":path}));
    }

    fn copy_recipe(app: &App, name: &str) {
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes").join("wood_rot.json"),
            app.root.join(name),
        )
        .unwrap();
    }

    #[test]
    fn new_tools_are_listed_with_strict_schemas_and_correct_read_only_hints() {
        let tools = tool_definitions();
        for (name, read_only) in [
            ("get_run_panel", true),
            ("get_scene_chemistry", true),
            ("set_scene_chemistry", false),
            ("inspect_creatures", true),
        ] {
            let definition = tools
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("{name} missing"));
            assert_eq!(definition["annotations"]["readOnlyHint"], read_only, "{name}");
            assert_eq!(definition["inputSchema"]["additionalProperties"], false, "{name}");
        }
        assert!(validate_tool_arguments("get_run_panel", &json!({})).is_ok());
        assert!(validate_tool_arguments("get_run_panel", &json!({"x":1})).is_err());
        assert!(validate_tool_arguments("set_scene_chemistry", &json!({}))
            .unwrap_err()
            .contains("scene_path"));
        assert!(validate_tool_arguments(
            "set_scene_chemistry",
            &json!({"scene_path":"a.vxscene","interval_s":-1.0})
        )
        .is_err());
        assert!(validate_tool_arguments(
            "set_scene_chemistry",
            &json!({"scene_path":"a.vxscene","enabled":"yes"})
        )
        .is_err());
        assert!(validate_tool_arguments(
            "inspect_creatures",
            &json!({"scene_path":"a.vxscene","extra":1})
        )
        .is_err());
    }

    #[test]
    fn chemistry_tools_write_the_same_settings_the_window_reads() {
        let app = app();
        new_scene(&app, "world.vxscene");
        copy_recipe(&app, "wood_rot.json");

        // Defaults for a scene without a settings file.
        let before = ok(&app, "get_scene_chemistry", json!({"scene_path":"world.vxscene"}));
        assert_eq!(before["settings"]["enabled"], false);
        assert_eq!(before["settings"]["interval_s"], 0.0);
        assert!(before["settings"]["recipe_file"].is_null());
        assert!(
            !app.root.join("world.chemistry.json").exists(),
            "reading must not create the file"
        );

        let after = ok(
            &app,
            "set_scene_chemistry",
            json!({"scene_path":"world.vxscene","enabled":true,"interval_s":3600.0,"recipe_path":"wood_rot.json"}),
        );
        assert_eq!(after["settings"]["enabled"], true);
        assert_eq!(after["settings"]["recipe_file"], "wood_rot.json");
        assert_eq!(after["recipe_count"], 1);
        assert!(after["recipe_error"].is_null());
        assert_eq!(after["settings_file"], "world.chemistry.json");

        // The editor's own loader reads exactly what the tool wrote, and
        // applying it configures a scene through the engine hooks.
        let scene_path = app.root.join("world.vxscene");
        let loaded = ChemistrySettings::load(&scene_path).unwrap();
        assert!(loaded.enabled && loaded.interval_s == 3600.0);
        let document: SceneDocument = load_document(&scene_path).unwrap();
        let mut scene = document.build_scene(&scene_path).unwrap();
        loaded.apply(&mut scene, Some(&scene_path)).unwrap();
        assert!(scene.chemistry_enabled);

        // A partial change keeps the rest.
        let changed = ok(
            &app,
            "set_scene_chemistry",
            json!({"scene_path":"world.vxscene","enabled":false}),
        );
        assert_eq!(changed["settings"]["enabled"], false);
        assert_eq!(changed["settings"]["interval_s"], 3600.0);
        assert_eq!(changed["settings"]["recipe_file"], "wood_rot.json");
        let builtin = ok(
            &app,
            "set_scene_chemistry",
            json!({"scene_path":"world.vxscene","use_builtin_recipes":true}),
        );
        assert!(builtin["settings"]["recipe_file"].is_null());
        assert!(builtin["recipe_count"].is_null());

        // It works inside a batch like the other file tools.
        let batch = ok(
            &app,
            "batch_actions",
            json!({"actions":[
                {"id":"on","name":"set_scene_chemistry","arguments":{"scene_path":"world.vxscene","enabled":true}},
                {"id":"read","name":"get_scene_chemistry","arguments":{"scene_path":"world.vxscene"}}
            ]}),
        );
        assert_eq!(batch["results"][1]["result"]["settings"]["enabled"], true);
        let _ = fs::remove_dir_all(app.root);
    }

    #[test]
    fn chemistry_errors_are_reported_and_leave_the_saved_settings_alone() {
        let app = app();
        new_scene(&app, "world.vxscene");
        copy_recipe(&app, "wood_rot.json");
        ok(
            &app,
            "set_scene_chemistry",
            json!({"scene_path":"world.vxscene","enabled":true,"interval_s":60.0}),
        );
        let settings_file = app.root.join("world.chemistry.json");
        let saved = fs::read_to_string(&settings_file).unwrap();

        fs::write(app.root.join("broken.json"), "{ not json").unwrap();
        fs::write(app.root.join("notes.txt"), "hello").unwrap();
        let cases: [(Value, &str); 7] = [
            (json!({"scene_path":"world.vxscene"}), "at least one"),
            (
                json!({"scene_path":"world.vxscene","recipe_path":"wood_rot.json","use_builtin_recipes":true}),
                "mutually exclusive",
            ),
            (json!({"scene_path":"world.vxscene","recipe_path":"missing.json"}), "resolve"),
            (json!({"scene_path":"world.vxscene","recipe_path":"broken.json"}), "broken.json"),
            (json!({"scene_path":"world.vxscene","recipe_path":"notes.txt"}), ".json"),
            (json!({"scene_path":"missing.vxscene","enabled":true}), "resolve"),
            (json!({"scene_path":"../outside.vxscene","enabled":true}), "outside"),
        ];
        for (arguments, expected) in cases {
            let message = error_text(&app, "set_scene_chemistry", arguments.clone());
            assert!(message.contains(expected), "{arguments}: {message}");
            assert_eq!(
                fs::read_to_string(&settings_file).unwrap(),
                saved,
                "{arguments} changed the file"
            );
        }
        ok(&app, "create_model", json!({"output_path":"m.vxm"}));
        assert!(
            error_text(&app, "set_scene_chemistry", json!({"scene_path":"m.vxm","enabled":true}))
                .contains(".vxscene")
        );

        // A damaged settings file is an error, not silently replaced.
        fs::write(&settings_file, "garbage").unwrap();
        assert!(error_text(&app, "get_scene_chemistry", json!({"scene_path":"world.vxscene"}))
            .contains("chemistry.json"));
        assert!(error_text(
            &app,
            "set_scene_chemistry",
            json!({"scene_path":"world.vxscene","enabled":false})
        )
        .contains("chemistry.json"));
        assert_eq!(fs::read_to_string(&settings_file).unwrap(), "garbage");

        // A recipe that disappears later is reported by the read tool, not a panic.
        fs::write(
            &settings_file,
            r#"{"version":1,"enabled":true,"interval_s":1.0,"recipe_file":"gone.json"}"#,
        )
        .unwrap();
        let reported = ok(&app, "get_scene_chemistry", json!({"scene_path":"world.vxscene"}));
        assert!(reported["recipe_error"].as_str().unwrap().contains("gone.json"));
        let _ = fs::remove_dir_all(app.root);
    }

    fn rigged_creature_scene(app: &App) {
        let source = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], 1))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], 1))
            .build()
            .unwrap();
        let mut model = VoxelModelDocument::from_scene("rigged", &source);
        let mut skeleton = SkeletonDocument::with_root_bone("root", DVec3::ZERO);
        let limb = skeleton
            .add_bone("limb", Some(0), DVec3::new(0.01, 0.0, 0.0))
            .unwrap();
        for voxel_index in 0..model.voxels.len() {
            skeleton.voxel_weights.push(VoxelBoneWeightDocument {
                voxel_index: voxel_index as u32,
                bone_index: limb as u32,
                weight: 1.0,
            });
        }
        let clip = skeleton.add_animation_clip("bend", 1.0).unwrap();
        for (time_s, z) in [(0.0, 0.0), (1.0, 0.01)] {
            skeleton
                .set_keyframe(
                    clip,
                    limb,
                    BoneKeyframeDocument {
                        time_s,
                        translation_m: DVec3::new(0.01, 0.0, z),
                        rotation: Quat::IDENTITY,
                    },
                )
                .unwrap();
        }
        model.skeleton = Some(skeleton);
        save_document(&app.root.join("rigged.vxm"), &model).unwrap();
        let mut document = SceneDocument::new("creature scene", SceneConfig::default()).unwrap();
        document
            .add_model(ModelInstanceDocument {
                asset_path: PathBuf::from("rigged.vxm"),
                translation_m: DVec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .unwrap();
        save_document(&app.root.join("zoo.vxscene"), &document).unwrap();
    }

    #[test]
    fn inspect_creatures_reports_none_without_a_file_and_the_real_ledger_and_sensors_with_one() {
        use voxel_physics_lab::creature_inspector::CreatureFile as File;
        let app = app();
        rigged_creature_scene(&app);
        let none = ok(&app, "inspect_creatures", json!({"scene_path":"zoo.vxscene"}));
        assert_eq!(none["creature_count"], 0);
        assert_eq!(none["creatures"], json!([]));

        let scene_path = app.root.join("zoo.vxscene");
        let document: SceneDocument = load_document(&scene_path).unwrap();
        let scene = document.build_scene(&scene_path).unwrap();
        let layout = SensorLayout::new(&scene.skeletal_actuators[0], vec![0]).unwrap();
        let organs = OrganParams {
            store_capacity_j: 80.0,
            birth_store_fraction: 0.25,
            stomach_capacity_j: 20.0,
            digestion_power_w: 2.0,
            absorption_efficiency: 0.8,
            muscle_efficiency: 0.25,
            basal_power_w: 0.0,
            peak_mechanical_power_w: 1.0,
            weak_fraction: 0.1,
            exhaustion_fraction: 0.0,
        };
        let genome = genome_for_body("rigged", &layout, organs, 3, 6, 1).unwrap();
        let file = File {
            version: CREATURE_FILE_VERSION,
            creatures: vec![CreatureSpec {
                name: "Ant".into(),
                actuator_index: 0,
                genome,
                learned: None,
            }],
        };
        fs::write(File::sidecar_path(&scene_path), serde_json::to_string(&file).unwrap()).unwrap();

        let result = ok(&app, "inspect_creatures", json!({"scene_path":"zoo.vxscene"}));
        assert_eq!(result["creature_count"], 1);
        let ant = &result["creatures"][0];
        assert_eq!(ant["name"], "Ant");
        assert_eq!(ant["ledger"]["stored_energy_j"], 20.0);
        assert_eq!(ant["ledger"]["store_capacity_j"], 80.0);
        assert_eq!(ant["ledger"]["work_j"], 0.0);
        assert_eq!(ant["ledger"]["heat_j"], 0.0);
        let sensors = ant["sensors"].as_array().unwrap();
        assert!(sensors.len() > 10);
        assert!(sensors.iter().all(|s| s["id"].is_string() && s["unit"].is_string()));
        let energy = sensors.iter().find(|s| s["id"] == "stored_energy").unwrap();
        assert_eq!(
            (energy["value"].as_f64(), energy["unit"].as_str()),
            (Some(0.25), Some("1"))
        );
        assert!(sensors.iter().any(|s| s["unit"] == "m") && sensors.iter().any(|s| s["unit"] == "N"));
        assert_eq!(ant["sensor_cost"]["this_read"]["reads"], 1);
        assert!(ant["network"]["cost"].is_null());

        // Inspecting twice gives the same answer: it counts nothing.
        let again = ok(&app, "inspect_creatures", json!({"scene_path":"zoo.vxscene"}));
        assert_eq!(again["creatures"], result["creatures"]);

        // A creature that does not fit the scene is an error naming it.
        let mut wrong = file;
        wrong.creatures[0].actuator_index = 4;
        fs::write(File::sidecar_path(&scene_path), serde_json::to_string(&wrong).unwrap()).unwrap();
        assert!(
            error_text(&app, "inspect_creatures", json!({"scene_path":"zoo.vxscene"})).contains("Ant")
        );
        let _ = fs::remove_dir_all(app.root);
    }

    #[test]
    fn get_run_panel_reads_what_the_editor_publishes() {
        let root = PathBuf::from(format!(
            "E:/voxel-editor-mcp-runpanel-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let server = voxel_physics_lab::runtime_metrics::start_server(&root).unwrap();
        let app = App { root: root.clone() };

        let mut panel = RunPanel::new();
        for (revision, cost) in [(1, 1.0), (2, 2.0), (3, 6.5)] {
            panel.record(7, revision, Some(cost));
        }
        let snapshot = panel.snapshot(RunPanelInput {
            state: RunState::Run,
            requested_speed: 1.0,
            measured_speed: Some(0.98),
            backlog_s: Some(0.02),
            active_mechanical_revs: Some(2048),
            solver_mode: Some(SolverMode::CellRotations),
        });
        voxel_physics_lab::runtime_metrics::publish("run_panel", snapshot.to_json());

        let result = call(&app, "get_run_panel", json!({}));
        assert_ne!(result["isError"], true, "{}", result["content"][0]["text"]);
        let data = &result["structuredContent"]["run_panel"];
        assert_eq!(data, &snapshot.to_json(), "MCP returns exactly the window's values");
        assert_eq!(data["state"], "Run");
        assert_eq!(data["worker_cost_ms_per_10ms"]["max"], 6.5);
        assert_eq!(data["worker_cost_ms_per_10ms"]["p50"], 2.0);
        assert_eq!(data["active_mechanical_revs"], 2048);
        assert_eq!(data["solver_mode"], "cell_rotations");
        assert!(result["structuredContent"]["age_ms"].is_u64());

        // A newer publication replaces it.
        panel.record(7, 4, Some(0.5));
        let paused = panel.snapshot(RunPanelInput {
            state: RunState::Pause,
            requested_speed: 1.0,
            measured_speed: None,
            backlog_s: None,
            active_mechanical_revs: Some(2048),
            solver_mode: Some(SolverMode::TranslationOnly),
        });
        voxel_physics_lab::runtime_metrics::publish("run_panel", paused.to_json());
        let latest = ok(&app, "get_run_panel", json!({}));
        assert_eq!(latest["run_panel"]["state"], "Pause");
        assert_eq!(latest["run_panel"]["solver_mode"], "translation_only");
        assert!(latest["run_panel"]["measured_world_speed"].is_null());

        // With no runtime the tool errors clearly instead of inventing values.
        drop(server);
        let missing = app_without_runtime();
        let text = error_text(&missing, "get_run_panel", json!({}));
        assert!(!text.is_empty());
        let _ = fs::remove_dir_all(&missing.root);
        let _ = fs::remove_dir_all(root);
    }

    fn app_without_runtime() -> App {
        app()
    }
}
