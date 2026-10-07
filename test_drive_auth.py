#!/usr/bin/env python3
"""
Test script to verify Google Drive API authentication
"""
import os
import sys

def test_drive_auth():
    try:
        # Add the app directory to Python path
        sys.path.insert(0, './app')
        
        # Import and test the drive client
        from drive_client import _get_service
        
        print("Testing Google Drive authentication...")
        service = _get_service()
        print("✓ Successfully created Drive service client")
        
        # Test a simple API call
        print("Testing basic API call...")
        about = service.about().get(fields="user").execute()
        print(f"✓ Successfully authenticated as: {about.get('user', {}).get('emailAddress', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    success = test_drive_auth()
    sys.exit(0 if success else 1)