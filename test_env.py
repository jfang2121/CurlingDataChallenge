"""
Test script to verify CurlingEnv works with vectorization.
"""

import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import sys

from curling_env import CurlingEnv

def make_env(rank):
    """Factory function to create a single environment."""
    def _make():
        return CurlingEnv()
    return _make


def test_single_env():
    """Test that a single environment works correctly."""
    print("=" * 60)
    print("TEST 1: Single Environment")
    print("=" * 60)
    
    try:
        env = CurlingEnv()
        obs, info = env.reset()
        
        print(f"✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        
        # Take a few steps
        for step_idx in range(3):
            action = env.action_space.sample()
            result = env.step(action)
            
            if len(result) != 5:
                print(f"✗ step() returned {len(result)} values, expected 5")
                print(f"  step() should return: (obs, reward, terminated, truncated, info)")
                return False
            
            obs, reward, terminated, truncated, info = result
            
            # Check reward is scalar
            if not isinstance(reward, (int, float, np.number)):
                print(f"✗ Reward is not scalar: {type(reward)} = {reward}")
                return False
            
            print(f"  Step {step_idx + 1}: reward={reward:.3f}, terminated={terminated}")
            
            if terminated:
                break
        
        env.close()
        print("✓ Single environment test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Single environment test FAILED")
        print(f"  Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_vectorized_env():
    """Test that the environment works with vectorization."""
    print("=" * 60)
    print("TEST 2: Vectorized Environment")
    print("=" * 60)
    
    try:
        num_envs = 4
        
        # Try to create vectorized environment
        env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
        print(f"✓ Created {num_envs} vectorized environments")
        
        # Reset
        obs, infos = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Expected shape: ({num_envs}, {env.single_observation_space.shape[0]})")
        
        if obs.shape != (num_envs, env.single_observation_space.shape[0]):
            print(f"✗ Observation shape mismatch!")
            return False
        
        # Take a few steps
        for step_idx in range(3):
            actions = np.array([env.single_action_space.sample() for _ in range(num_envs)])
            
            print(actions)
            result = env.step(actions)
            
            if len(result) != 5:
                print(f"✗ step() returned {len(result)} values, expected 5")
                return False
            
            obs, rewards, terminateds, truncateds, infos = result
            
            # Check shapes
            if rewards.shape != (num_envs,):
                print(f"✗ Rewards shape incorrect: {rewards.shape}, expected ({num_envs},)")
                return False
            
            if obs.shape != (num_envs, env.single_observation_space.shape[0]):
                print(f"✗ Observation shape incorrect: {obs.shape}")
                return False
            
            print(f"  Step {step_idx + 1}:")
            print(f"    Rewards: {rewards}")
            print(f"    Any done: {np.any(terminateds | truncateds)}")
            
            if np.all(terminateds | truncateds):
                print("    All environments terminated")
                break
        
        env.close()
        print("✓ Vectorized environment test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Vectorized environment test FAILED")
        print(f"  Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch processing with PyTorch (simulated)."""
    print("=" * 60)
    print("TEST 3: Batch Processing Compatibility")
    print("=" * 60)
    
    try:
        import torch
        
        num_envs = 4
        env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
        obs, _ = env.reset()
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs)
        print(f"✓ Converted observations to PyTorch tensor")
        print(f"  Tensor shape: {obs_tensor.shape}")
        print(f"  Tensor dtype: {obs_tensor.dtype}")
        
        # Simulate batch action sampling
        actions = np.random.uniform(
            low=env.single_action_space.low,
            high=env.single_action_space.high,
            size=(num_envs, env.single_action_space.shape[0])
        )
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        print(f"✓ Batch processing works correctly")
        print(f"  Stepped {num_envs} environments in one call")
        print(f"  Rewards shape: {rewards.shape}")
        
        env.close()
        print("✓ Batch processing test PASSED\n")
        return True
        
    except ImportError:
        print("⊘ PyTorch not available, skipping test\n")
        return True
    except Exception as e:
        print(f"✗ Batch processing test FAILED")
        print(f"  Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CURLING ENVIRONMENT VECTORIZATION TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Single environment
    results.append(("Single Env", test_single_env()))
    
    # Test 2: Vectorized environment
    results.append(("Vectorized Env", test_vectorized_env()))
    
    # Test 3: Batch processing
    results.append(("Batch Processing", test_batch_processing()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Your environment is vectorization-ready.\n")
        return 0
    else:
        print("\n❌ Some tests failed. See errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())