use std::env;

fn main() {
    // Priority order for Python resolution:
    //   1. PYO3_PYTHON (explicit override — set this in Docker or CI)
    //   2. CONDA_PREFIX (backwards compat for local dev with conda)
    //   3. System python3 (fallback — works in Docker)

    if let Ok(pyo3_python) = env::var("PYO3_PYTHON") {
        // Explicit override — Docker sets this to "python3"
        println!("cargo:warning=Using PYO3_PYTHON: {}", pyo3_python);
    } else if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        // Local conda dev environment
        println!("cargo:warning=Using Conda environment: {}", conda_prefix);
        println!("cargo:rustc-link-search=native={}/lib", conda_prefix);
        let python_exe = format!("{}/bin/python", conda_prefix);
        println!("cargo:rustc-env=PYO3_PYTHON={}", python_exe);
    } else {
        // Docker / CI / system Python — PyO3 will find python3 automatically
        println!("cargo:warning=No CONDA_PREFIX or PYO3_PYTHON set. Using system python3.");
    }

    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
}
