use voxel_physics_lab::experiments;
fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let stages = if which == "all" {
        vec!["tension", "compression", "bending", "wedge", "knife"]
    } else {
        vec![which.as_str()]
    };
    for stage in stages {
        let result = pool.install(|| match stage {
            "tension" => experiments::tension(),
            "compression" => experiments::compression(),
            "bending" => experiments::bending(),
            "wedge" => experiments::wedge_split(),
            "knife" => experiments::knife_cut(),
            _ => panic!("Unknown or blocked fixture: {stage}"),
        });
        println!(
            "{} {}: {}",
            if result.passed { "PASS" } else { "FAIL" },
            result.name,
            result.detail
        );
        if !result.passed {
            eprintln!("Acceptance gate closed. Later experiments are not run.");
            std::process::exit(1);
        }
    }
}
