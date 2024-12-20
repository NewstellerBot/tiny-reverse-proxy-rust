use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tiny_reverse_proxy_rust::thread_pool::ThreadPool;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threadpool_executes_jobs() {
        let pool = ThreadPool::new(4);
        let result = Arc::new(Mutex::new(0));

        for i in 1..=10 {
            let result_clone = Arc::clone(&result);
            pool.execute(move || {
                let mut num = result_clone.lock().unwrap();
                *num += i;
            });
        }

        // Wait for threads to complete
        thread::sleep(Duration::from_millis(100));

        assert_eq!(*result.lock().unwrap(), 55); // Sum of 1..=10
    }

    #[test]
    fn threadpool_handles_multiple_threads() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(Mutex::new(0));

        for _ in 0..8 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                let mut count = counter_clone.lock().unwrap();
                *count += 1;
                thread::sleep(Duration::from_millis(50));
            });
        }

        // Wait for threads to complete
        thread::sleep(Duration::from_millis(500));

        assert_eq!(*counter.lock().unwrap(), 8);
    }

    #[test]
    fn threadpool_shuts_down_gracefully() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(Mutex::new(0));

        for _ in 0..4 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                let mut count = counter_clone.lock().unwrap();
                *count += 1;
                thread::sleep(Duration::from_millis(50));
            });
        }

        // Drop the pool and ensure no panic occurs
        drop(pool);
        assert!(true); // If the test reaches here without panic, it's successful
    }

    #[test]
    fn threadpool_executes_jobs_concurrently() {
        let pool = ThreadPool::new(4);
        let start_time = std::time::Instant::now();
        let counter = Arc::new(Mutex::new(0));

        for _ in 0..4 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                thread::sleep(Duration::from_millis(100));
                let mut count = counter_clone.lock().unwrap();
                *count += 1;
            });
        }

        thread::sleep(Duration::from_millis(200)); // Wait for tasks to complete
        let elapsed_time = start_time.elapsed();

        assert!(elapsed_time < Duration::from_millis(500)); // Ensure concurrent execution
        assert_eq!(*counter.lock().unwrap(), 4);
    }
}
