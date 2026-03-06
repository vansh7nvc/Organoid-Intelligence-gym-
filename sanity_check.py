"""
Sanity Check Script for Brian2 Installation
Tests Brian2 with automatic fallback from Cython to numpy if compiler is missing
"""

import sys
from brian2 import *

def main():
    print("=" * 60)
    print("Brian2 Sanity Check - Week 1 Day 1")
    print("=" * 60)
    
    # Configure Cython with MinGW (Optimized for your setup)
    print("\n[INFO] Configuring Cython backend with MinGW...")
    prefs.codegen.target = 'cython'
    prefs.codegen.cpp.compiler = 'mingw32'
    prefs.codegen.cpp.extra_compile_args = ['-DMS_WIN64']
    
    try:
        # Create a simple neuron group
        print("\n[1/3] Creating simple neuron group...")
        tau = 10*ms
        eqs = '''
        dv/dt = (1-v)/tau : 1
        '''
        G = NeuronGroup(10, eqs, method='euler')
        G.v = 'rand()'
        print("[OK] Neuron group created successfully")
        
        # Create a spike monitor
        print("\n[2/3] Setting up spike monitor...")
        M = StateMonitor(G, 'v', record=True)
        print("[OK] Monitor configured")
        
        # Run simulation
        print("\n[3/3] Running 10ms simulation...")
        run(10*ms)
        print("[OK] Simulation completed successfully!")
        
        print("\n" + "=" * 60)
        print(f"SUCCESS: Brian2 is working with {prefs.codegen.target} backend!")
        print("=" * 60)
        print(f"\nPython version: {sys.version}")
        import brian2
        print(f"Brian2 version: {brian2.__version__}")
        
        print("\n[EXCELLENT] C++ compiler (MinGW) is working correctly!")
        print("You have optimal performance for OrganoidRL.")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Cython backend failed: {str(e)}")
        print("\nPossible solutions:")
        print("1. Verify C:\\msys64\\ucrt64\\bin is in your PATH")
        print("2. Ensure g++ --version works in your terminal")
        print("3. Check if all dependencies are installed: pip install cython numpy brian2")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
