#!/usr/bin/env python3
"""
Script to create simple test models for inference engine testing.
This creates minimal models that can be used to test Candle and ORT engines.
"""

import torch
import torch.nn as nn
import numpy as np
import os

def create_simple_linear_model():
    """Create a simple linear model for testing"""
    print("📦 Creating simple linear model...")
    
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            # Set deterministic weights for testing
            with torch.no_grad():
                self.linear.weight.fill_(0.5)
                self.linear.bias.fill_(0.1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleLinear()
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, 2)
    
    # Export to ONNX
    onnx_path = "simple_linear.onnx"
    torch.onnx.export(
        model,
        test_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Created ONNX model: {onnx_path}")
    
    # Save as SafeTensors for Candle
    try:
        from safetensors.torch import save_file
        state_dict = model.state_dict()
        safetensors_path = "simple_linear.safetensors"
        save_file(state_dict, safetensors_path)
        print(f"✅ Created SafeTensors model: {safetensors_path}")
    except ImportError:
        print("⚠️  SafeTensors not available, skipping SafeTensors export")
        print("   Install with: pip install safetensors")
    
    # Test the model
    with torch.no_grad():
        output = model(test_input)
        print(f"📊 Test input: {test_input.numpy()}")
        print(f"📊 Test output: {output.numpy()}")
    
    return onnx_path

def create_simple_classifier():
    """Create a simple classifier for testing"""
    print("\n🧠 Creating simple classifier...")
    
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 3)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return self.softmax(x)
    
    model = SimpleClassifier()
    model.eval()
    
    # Create test input (batch_size=1, features=4)
    test_input = torch.randn(1, 4)
    
    # Export to ONNX
    onnx_path = "simple_classifier.onnx"
    torch.onnx.export(
        model,
        test_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Created ONNX classifier: {onnx_path}")
    
    # Test the model
    with torch.no_grad():
        output = model(test_input)
        print(f"📊 Test input shape: {test_input.shape}")
        print(f"📊 Test output shape: {output.shape}")
        print(f"📊 Output probabilities: {output.numpy()}")
    
    return onnx_path

def create_test_data():
    """Create test data files"""
    print("\n📊 Creating test data files...")
    
    # Create test input data
    test_data = {
        "linear_input": np.array([[1.0, 2.0]], dtype=np.float32),
        "classifier_input": np.array([[0.5, -0.3, 1.2, -0.8]], dtype=np.float32),
        "batch_input": np.random.randn(4, 2).astype(np.float32)
    }
    
    for name, data in test_data.items():
        np.save(f"{name}.npy", data)
        print(f"📁 Created test data: {name}.npy (shape: {data.shape})")

def main():
    print("🚀 Creating Test Models for Inference Engines")
    print("==============================================")
    
    # Check PyTorch availability
    print(f"🔧 PyTorch version: {torch.__version__}")
    
    # Create models
    try:
        linear_model = create_simple_linear_model()
        classifier_model = create_simple_classifier()
        create_test_data()
        
        print("\n✅ Model Creation Summary:")
        print("==========================")
        
        # List created files
        files = [
            "simple_linear.onnx",
            "simple_linear.safetensors", 
            "simple_classifier.onnx",
            "linear_input.npy",
            "classifier_input.npy",
            "batch_input.npy"
        ]
        
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"📁 {file} ({size} bytes)")
            else:
                print(f"❌ {file} (not created)")
        
        print("\n🎉 Test models created successfully!")
        print("💡 You can now run the Rust tests with these models.")
        
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        print("💡 Make sure you have PyTorch installed: pip install torch")

if __name__ == "__main__":
    main() 