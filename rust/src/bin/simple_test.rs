use std::collections::HashMap;

fn main() {
    println!("🚀 Simple Rust Test");
    println!("===================");
    
    // Test basic functionality
    let mut results = HashMap::new();
    results.insert("basic", true);
    
    // Test Candle compilation
    #[cfg(feature = "candle")]
    {
        println!("✅ Candle feature enabled");
        results.insert("candle", true);
    }
    
    #[cfg(not(feature = "candle"))]
    {
        println!("❌ Candle feature disabled");
        results.insert("candle", false);
    }
    
    // ORT is no longer supported
    println!("❌ ORT feature removed (no longer supported)");
    results.insert("ort", false);
    
    // Test Linfa compilation
    #[cfg(feature = "linfa")]
    {
        println!("✅ Linfa feature enabled");
        results.insert("linfa", true);
    }
    
    #[cfg(not(feature = "linfa"))]
    {
        println!("❌ Linfa feature disabled");
        results.insert("linfa", false);
    }
    
    // Summary
    let enabled = results.values().filter(|&&v| v).count();
    let total = results.len();
    
    println!("\n🏁 Feature Summary: {}/{} enabled", enabled, total);
    
    for (feature, enabled) in &results {
        let status = if *enabled { "✅" } else { "❌" };
        println!("{} {}", status, feature.to_uppercase());
    }
    
    println!("\n✅ Basic compilation test passed!");
} 