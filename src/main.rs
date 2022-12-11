#[cfg(feature = "intel-mkl-amd")]
use std::os::raw::c_int;
use std::time::Duration;

use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, LinalgScalar};
use ndarray::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use structopt::StructOpt;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "blis")]
extern crate blis_src;

#[cfg(any(feature = "intel-mkl", feature = "intel-mkl-amd"))]
extern crate intel_mkl_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

#[derive(StructOpt)]
#[structopt(name = "basic")]
struct Opts {
    /// Matrix dimensionality (N x N).
    #[structopt(short, long, default_value = "256")]
    dim: usize,
    /// Double precision (DGEMM).
    #[structopt(long)]
    dgemm: bool,
    /// The number of gemm iterations per thread.
    #[structopt(short, long, default_value = "1000")]
    iterations: usize,
    /// The number of benchmark threads.
    #[structopt(short, long, default_value = "1")]
    threads: usize,
}

struct BenchmarkStats {
    elapsed: Duration,
    flops: usize,
}

fn gemm_benchmark<A>(dim: usize, iterations: usize, threads: usize) -> BenchmarkStats
where
    A: LinalgScalar + Send + Sync,
{
    let one = A::one();
    let two = one + one;
    let point_five = one / two;

    let a1 = Array::<A, _>::ones([dim, dim].f());
    // println!("{:?}", a1::<f32, _>);
    
    let a2 = Array::<A, _>::from_elem([dim, dim], point_five);
    let a3 = Array::<A, _>::from_elem([dim, dim], one);

    let c_matrices: Vec<_> = std::iter::repeat(Array2::from_elem((dim, dim), one))
        .take(threads)
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();

    c_matrices.into_par_iter().for_each(|mut matrix_c| {
        for _ in 0..iterations {
            general_mat_mul(one, &a1, &a2, A::one(), &mut matrix_c);
        }
    });

    let elapsed = start.elapsed();

    BenchmarkStats {
        elapsed,
        flops: (dim.pow(3) * 2 * iterations * threads) + (dim.pow(2) * 2 * iterations * threads),
    }
}

fn main() {
    let opts: Opts = Opts::from_args();

    rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build_global()
        .unwrap();

    println!("Threads: {}", opts.threads);
    println!("Iterations per thread: {}", opts.iterations);
    println!("Matrix shape: {} x {}", opts.dim, opts.dim);

    let stats = if opts.dgemm {
        gemm_benchmark::<f64>(opts.dim, opts.iterations, opts.threads)
    } else {
        println!("benchmarking on f32.");
        gemm_benchmark::<f32>(opts.dim, opts.iterations, opts.threads)
    };

    println!(
        "GFLOPS: {:.2}",
        (stats.flops as f64 / stats.elapsed.as_secs_f64()) / 1000_000_000.
    );
}

#[cfg(feature = "intel-mkl-amd")]
#[allow(dead_code)]
#[no_mangle]
extern "C" fn mkl_serv_intel_cpu_true() -> c_int {
    1
}
