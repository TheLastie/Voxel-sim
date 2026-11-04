use super::{check, ContactIc0Input, CudaSolver, DevicePtr, Handle};
use glam::Vec3;
use std::ffi::c_void;

const IC0_PTX: &str = r#"
.visible .entry contact_ic0_level(
 .param .u64 inverse_ptr, .param .u64 factors_ptr,
 .param .u64 links_ptr, .param .u64 counts_ptr,
 .param .u64 rows_ptr, .param .u32 row_count,
 .param .u64 rhs_ptr, .param .u64 work_ptr, .param .u64 out_ptr,
 .param .u32 reverse
) {
 .reg .pred %p<16>; .reg .b32 %r<32>; .reg .b64 %rd<24>; .reg .f32 %f<12>;
 .local .align 4 .b8 value[12]; .local .align 4 .b8 result[12];
 ld.param.u64 %rd1,[inverse_ptr]; ld.param.u64 %rd2,[factors_ptr];
 ld.param.u64 %rd3,[links_ptr]; ld.param.u64 %rd4,[counts_ptr];
 ld.param.u64 %rd5,[rows_ptr]; ld.param.u32 %r1,[row_count];
 ld.param.u64 %rd6,[rhs_ptr]; ld.param.u64 %rd7,[work_ptr];
 ld.param.u64 %rd8,[out_ptr]; ld.param.u32 %r2,[reverse];
 mov.u32 %r3,%ctaid.x; mov.u32 %r4,%ntid.x; mov.u32 %r5,%tid.x;
 mad.lo.u32 %r0,%r3,%r4,%r5; setp.ge.u32 %p0,%r0,%r1; @%p0 bra DONE;
 mul.wide.u32 %rd9,%r0,4; add.u64 %rd10,%rd5,%rd9; ld.global.u32 %r6,[%rd10];
 mul.wide.u32 %rd11,%r6,12;
 setp.eq.u32 %p1,%r2,0;
 @%p1 add.u64 %rd12,%rd6,%rd11;
 @!%p1 add.u64 %rd12,%rd7,%rd11;
 mov.u32 %r7,0;
 INIT:
 setp.ge.u32 %p2,%r7,3; @%p2 bra LINKS_BEGIN;
 mul.wide.u32 %rd13,%r7,4; add.u64 %rd14,%rd12,%rd13; ld.global.f32 %f0,[%rd14];
 mov.u64 %rd15,value; add.u64 %rd15,%rd15,%rd13; st.local.f32 [%rd15],%f0;
 add.u32 %r7,%r7,1; bra INIT;
 LINKS_BEGIN:
 mul.wide.u32 %rd13,%r6,48; add.u64 %rd14,%rd3,%rd13;
 mul.wide.u32 %rd13,%r6,1; add.u64 %rd15,%rd4,%rd13; ld.global.u8 %r8,[%rd15];
 mov.u32 %r9,0;
 LINK_LOOP:
 setp.ge.u32 %p2,%r9,%r8; @%p2 bra DIAG_BEGIN;
 mul.wide.u32 %rd13,%r9,8; add.u64 %rd16,%rd14,%rd13;
 ld.global.u32 %r10,[%rd16]; ld.global.u32 %r11,[%rd16+4];
 mul.wide.u32 %rd17,%r10,36; add.u64 %rd17,%rd2,%rd17;
 mul.wide.u32 %rd18,%r11,12; setp.eq.u32 %p3,%r2,0;
 @%p3 add.u64 %rd18,%rd7,%rd18;
 @!%p3 add.u64 %rd18,%rd8,%rd18;
 mov.u32 %r12,0;
 COMP_LOOP:
 setp.ge.u32 %p4,%r12,3; @%p4 bra LINK_DONE;
 mov.f32 %f1,0f00000000; mov.u32 %r13,0;
 COL_LOOP:
 setp.ge.u32 %p5,%r13,3; @%p5 bra SUM_DONE;
 // matrix column-major; transpose the factor for the reverse sweep.
 setp.eq.u32 %p6,%r2,0;
 mad.lo.u32 %r14,%r13,3,%r12;
 mad.lo.u32 %r15,%r12,3,%r13;
 selp.u32 %r16,%r14,%r15,%p6;
 mul.wide.u32 %rd19,%r16,4; add.u64 %rd19,%rd17,%rd19; ld.global.f32 %f2,[%rd19];
 mul.wide.u32 %rd20,%r13,4; add.u64 %rd20,%rd18,%rd20; ld.global.f32 %f3,[%rd20];
 mul.rn.f32 %f4,%f2,%f3; add.rn.f32 %f1,%f1,%f4;
 add.u32 %r13,%r13,1; bra COL_LOOP;
 SUM_DONE:
 mul.wide.u32 %rd19,%r12,4; mov.u64 %rd20,value; add.u64 %rd20,%rd20,%rd19;
 ld.local.f32 %f5,[%rd20]; sub.rn.f32 %f5,%f5,%f1; st.local.f32 [%rd20],%f5;
 add.u32 %r12,%r12,1; bra COMP_LOOP;
 LINK_DONE:
 add.u32 %r9,%r9,1; bra LINK_LOOP;
 DIAG_BEGIN:
 mul.wide.u32 %rd17,%r6,36; add.u64 %rd17,%rd1,%rd17;
 mov.u32 %r12,0;
 DIAG_COMP:
 setp.ge.u32 %p4,%r12,3; @%p4 bra STORE_BEGIN;
 mov.f32 %f1,0f00000000; mov.u32 %r13,0;
 DIAG_COL:
 setp.ge.u32 %p5,%r13,3; @%p5 bra DIAG_SUM;
 setp.eq.u32 %p6,%r2,0;
 mad.lo.u32 %r14,%r13,3,%r12; mad.lo.u32 %r15,%r12,3,%r13;
 selp.u32 %r16,%r14,%r15,%p6;
 mul.wide.u32 %rd19,%r16,4; add.u64 %rd19,%rd17,%rd19; ld.global.f32 %f2,[%rd19];
 mul.wide.u32 %rd20,%r13,4; mov.u64 %rd21,value; add.u64 %rd21,%rd21,%rd20; ld.local.f32 %f3,[%rd21];
 mul.rn.f32 %f4,%f2,%f3; add.rn.f32 %f1,%f1,%f4;
 add.u32 %r13,%r13,1; bra DIAG_COL;
 DIAG_SUM:
 mul.wide.u32 %rd19,%r12,4; mov.u64 %rd20,result; add.u64 %rd20,%rd20,%rd19; st.local.f32 [%rd20],%f1;
 add.u32 %r12,%r12,1; bra DIAG_COMP;
 STORE_BEGIN:
 setp.eq.u32 %p1,%r2,0; @%p1 add.u64 %rd12,%rd7,%rd11; @!%p1 add.u64 %rd12,%rd8,%rd11;
 mov.u32 %r7,0;
 STORE_LOOP:
 setp.ge.u32 %p2,%r7,3; @%p2 bra DONE;
 mul.wide.u32 %rd13,%r7,4; mov.u64 %rd14,result; add.u64 %rd14,%rd14,%rd13; ld.local.f32 %f0,[%rd14];
 add.u64 %rd15,%rd12,%rd13; st.global.f32 [%rd15],%f0;
 add.u32 %r7,%r7,1; bra STORE_LOOP;
 DONE: ret;
}
"#;

struct Levels {
    offsets: Vec<u32>,
    rows: Vec<u32>,
}

pub(super) struct PreparedIc0 {
    inverse: DevicePtr,
    factors: DevicePtr,
    lower: DevicePtr,
    lower_counts: DevicePtr,
    upper: DevicePtr,
    upper_counts: DevicePtr,
    forward_rows: DevicePtr,
    reverse_rows: DevicePtr,
    forward: Levels,
    reverse: Levels,
    rhs: DevicePtr,
    work: DevicePtr,
    out: DevicePtr,
    host: Vec<f32>,
}

pub(super) fn level_counts(prepared: &PreparedIc0) -> [usize; 2] {
    [
        prepared.forward.offsets.len() - 1,
        prepared.reverse.offsets.len() - 1,
    ]
}

fn levels(
    count: usize,
    links: &[[[u32; 2]; 6]],
    counts: &[u8],
    name: &str,
) -> Result<Levels, String> {
    let mut dependent_offsets = vec![0usize; count + 1];
    let mut indegree = vec![0usize; count];
    for row in 0..count {
        for link in &links[row][..counts[row] as usize] {
            let neighbor = link[1] as usize;
            dependent_offsets[neighbor + 1] += 1;
            indegree[row] += 1;
        }
    }
    for node in 0..count {
        dependent_offsets[node + 1] += dependent_offsets[node];
    }
    let mut dependents = vec![0usize; dependent_offsets[count]];
    let mut cursor = dependent_offsets[..count].to_vec();
    for row in 0..count {
        for link in &links[row][..counts[row] as usize] {
            let neighbor = link[1] as usize;
            dependents[cursor[neighbor]] = row;
            cursor[neighbor] += 1;
        }
    }
    let mut frontier: Vec<usize> = (0..count).filter(|&row| indegree[row] == 0).collect();
    let mut level_offsets = vec![0];
    let mut rows = Vec::with_capacity(count);
    let mut done = 0;
    while !frontier.is_empty() {
        frontier.sort_unstable();
        let current = std::mem::take(&mut frontier);
        for &row in &current {
            rows.push(row as u32);
            done += 1;
        }
        for row in current {
            for &dependent in &dependents[dependent_offsets[row]..dependent_offsets[row + 1]] {
                indegree[dependent] -= 1;
                if indegree[dependent] == 0 {
                    frontier.push(dependent);
                }
            }
        }
        level_offsets.push(rows.len() as u32);
    }
    if done != count {
        return Err(format!("CUDA IC0 {name} dependencies contain a cycle"));
    }
    Ok(Levels {
        offsets: level_offsets,
        rows,
    })
}

pub(super) fn prepare(
    cuda: &mut CudaSolver,
    count: usize,
    input: ContactIc0Input<'_>,
) -> Result<PreparedIc0, String> {
    if input.inverse_diagonal.len() != count
        || input.lower_links.len() != count
        || input.lower_counts.len() != count
        || input.upper_links.len() != count
        || input.upper_counts.len() != count
    {
        return Err("CUDA IC0 arrays have inconsistent row lengths".into());
    }
    if input
        .inverse_diagonal
        .iter()
        .flatten()
        .chain(input.edge_factors.iter().flatten())
        .any(|x| !x.is_finite())
    {
        return Err("CUDA IC0 factors contain non-finite values".into());
    }
    for (name, links, counts) in [
        ("lower", input.lower_links, input.lower_counts),
        ("upper", input.upper_links, input.upper_counts),
    ] {
        for row in 0..count {
            if counts[row] as usize > 6 {
                return Err(format!("CUDA IC0 {name} row {row} exceeds six links"));
            }
            for &[edge, neighbor] in &links[row][..counts[row] as usize] {
                if edge as usize >= input.edge_factors.len()
                    || neighbor as usize >= count
                    || neighbor == row as u32
                {
                    return Err(format!(
                        "CUDA IC0 {name} row {row} has an out-of-range or self link"
                    ));
                }
            }
        }
    }
    let forward = levels(count, input.lower_links, input.lower_counts, "lower")?;
    let reverse = levels(count, input.upper_links, input.upper_counts, "upper")?;
    cuda.ensure_ic0_kernel()?;
    let inverse = cuda.upload("contact_ic0_inverse", input.inverse_diagonal)?;
    let factors = cuda.upload("contact_ic0_factors", input.edge_factors)?;
    let lower = cuda.upload("contact_ic0_lower", input.lower_links)?;
    let lower_counts = cuda.upload("contact_ic0_lower_counts", input.lower_counts)?;
    let upper = cuda.upload("contact_ic0_upper", input.upper_links)?;
    let upper_counts = cuda.upload("contact_ic0_upper_counts", input.upper_counts)?;
    let forward_rows = cuda.upload("contact_ic0_forward_rows", &forward.rows)?;
    let reverse_rows = cuda.upload("contact_ic0_reverse_rows", &reverse.rows)?;
    let bytes = count * 12;
    let rhs = cuda.buffer("contact_ic0_rhs", bytes)?;
    let work = cuda.buffer("contact_ic0_work", bytes)?;
    let out = cuda.buffer("contact_ic0_out", bytes)?;
    Ok(PreparedIc0 {
        inverse,
        factors,
        lower,
        lower_counts,
        upper,
        upper_counts,
        forward_rows,
        reverse_rows,
        forward,
        reverse,
        rhs,
        work,
        out,
        host: vec![0.; count * 3],
    })
}

pub(super) fn apply(
    cuda: &mut CudaSolver,
    count: usize,
    p: &mut PreparedIc0,
    rhs: &[Vec3],
    out: &mut [Vec3],
) -> Result<(), String> {
    if rhs.len() != count || out.len() != count {
        return Err(format!(
            "CUDA IC0 expected {count} rows, got {} and {}",
            rhs.len(),
            out.len()
        ));
    }
    p.host.clear();
    p.host.extend(rhs.iter().flat_map(|v| [v.x, v.y, v.z]));
    if p.host.iter().any(|v| !v.is_finite()) {
        return Err("CUDA IC0 rhs contains non-finite values".into());
    }
    cuda.upload("contact_ic0_rhs", &p.host)?;
    launch_levels(cuda, p, false)?;
    launch_levels(cuda, p, true)?;
    cuda.download(p.host.as_mut_ptr().cast(), p.out, count * 12)?;
    if p.host.iter().any(|v| !v.is_finite()) {
        return Err("CUDA IC0 produced non-finite output".into());
    }
    for (dst, xyz) in out.iter_mut().zip(p.host.chunks_exact(3)) {
        *dst = Vec3::new(xyz[0], xyz[1], xyz[2]);
    }
    Ok(())
}

fn launch_levels(cuda: &mut CudaSolver, p: &PreparedIc0, reverse: bool) -> Result<(), String> {
    let levels = if reverse { &p.reverse } else { &p.forward };
    let rows = if reverse {
        p.reverse_rows
    } else {
        p.forward_rows
    };
    let links = if reverse { p.upper } else { p.lower };
    let counts = if reverse {
        p.upper_counts
    } else {
        p.lower_counts
    };
    for pair in levels.offsets.windows(2) {
        let start = pair[0];
        let n = pair[1] - start;
        if n == 0 {
            continue;
        }
        cuda.launch_ic0_level(
            p.inverse, p.factors, links, counts, rows, start, n, p.rhs, p.work, p.out, reverse,
        )?;
    }
    Ok(())
}

impl CudaSolver {
    fn ensure_ic0_kernel(&mut self) -> Result<Handle, String> {
        if let Some(kernel) = self.ic0_kernel {
            return Ok(kernel);
        }
        let mut ptx = b".version 7.0\n.target sm_75\n.address_size 64\n".to_vec();
        ptx.extend_from_slice(IC0_PTX.as_bytes());
        ptx.push(0);
        let mut module = 0;
        let status = unsafe { (self.driver.module_load_data)(&mut module, ptx.as_ptr().cast()) };
        check(status, "cuModuleLoadData(contact IC0 PTX)")?;
        let mut kernel = 0;
        check(
            unsafe {
                (self.driver.module_get_function)(
                    &mut kernel,
                    module,
                    b"contact_ic0_level\0".as_ptr().cast(),
                )
            },
            "cuModuleGetFunction(contact_ic0_level)",
        )?;
        self.ic0_module = Some(module);
        self.ic0_kernel = Some(kernel);
        Ok(kernel)
    }
    fn launch_ic0_level(
        &self,
        inverse: DevicePtr,
        factors: DevicePtr,
        links: DevicePtr,
        counts: DevicePtr,
        rows: DevicePtr,
        start: u32,
        n: u32,
        rhs: DevicePtr,
        work: DevicePtr,
        out: DevicePtr,
        reverse: bool,
    ) -> Result<(), String> {
        let kernel = self
            .ic0_kernel
            .ok_or_else(|| "CUDA IC0 kernel is not loaded".to_string())?;
        let mut ptrs = [
            inverse,
            factors,
            links,
            counts,
            rows + u64::from(start) * 4,
            rhs,
            work,
            out,
        ];
        let mut narg = n;
        let mut mode = u32::from(reverse);
        let mut params: Vec<*mut c_void> = ptrs
            .iter_mut()
            .map(|p| (p as *mut DevicePtr).cast())
            .collect();
        params.insert(5, (&mut narg as *mut u32).cast());
        params.push((&mut mode as *mut u32).cast());
        check(
            unsafe {
                (self.driver.launch_kernel)(
                    kernel,
                    n.div_ceil(128),
                    1,
                    1,
                    128,
                    1,
                    1,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel(contact_ic0_level)",
        )
    }
}
