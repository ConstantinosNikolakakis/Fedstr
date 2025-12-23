use std::env;

fn main() {
    // Set Python configuration
    // Users should set CONDA_PREFIX or PYTHON_SYS_EXECUTABLE environment variable
    
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        println!("cargo:warning=Using Conda environment: {}", conda_prefix);
        println!("cargo:rustc-link-search=native={}/lib", conda_prefix);
        
        // Set Python executable
        let python_exe = format!("{}/bin/python", conda_prefix);
        println!("cargo:rustc-env=PYO3_PYTHON={}", python_exe);
    } else {
        println!("cargo:warning=CONDA_PREFIX not set. Using system Python.");
        println!("cargo:warning=Set CONDA_PREFIX to your conda environment path for consistent builds.");
    }
    
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
}
