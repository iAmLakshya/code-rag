/**
 * User profile component displaying user information.
 */

import React, { useEffect, useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { fetchUserProfile } from '../hooks/useApi';

interface User {
  id: number;
  email: string;
  username: string;
  created_at: string;
  is_active: boolean;
}

interface ProfileState {
  user: User | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * UserProfile displays the current user's information.
 *
 * Features:
 * - Fetches and displays user data
 * - Shows loading state
 * - Handles errors gracefully
 * - Provides logout functionality
 */
export const UserProfile: React.FC = () => {
  const { user: authUser, logout, isAuthenticated } = useAuth();
  const [state, setState] = useState<ProfileState>({
    user: null,
    isLoading: true,
    error: null,
  });

  useEffect(() => {
    if (!isAuthenticated) {
      setState(prev => ({ ...prev, isLoading: false }));
      return;
    }

    loadProfile();
  }, [isAuthenticated]);

  /**
   * Load user profile from API.
   */
  const loadProfile = async () => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));
      const profile = await fetchUserProfile();
      setState({ user: profile, isLoading: false, error: null });
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load profile',
      }));
    }
  };

  /**
   * Handle logout button click.
   */
  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  /**
   * Format date for display.
   */
  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  if (!isAuthenticated) {
    return (
      <div className="profile-container">
        <p>Please log in to view your profile.</p>
      </div>
    );
  }

  if (state.isLoading) {
    return (
      <div className="profile-container">
        <div className="loading-spinner">Loading profile...</div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="profile-container">
        <div className="error-message">{state.error}</div>
        <button onClick={loadProfile}>Retry</button>
      </div>
    );
  }

  if (!state.user) {
    return null;
  }

  return (
    <div className="profile-container">
      <div className="profile-header">
        <h2>User Profile</h2>
        <button onClick={handleLogout} className="logout-button">
          Logout
        </button>
      </div>

      <div className="profile-info">
        <div className="info-row">
          <span className="label">Username:</span>
          <span className="value">{state.user.username}</span>
        </div>

        <div className="info-row">
          <span className="label">Email:</span>
          <span className="value">{state.user.email}</span>
        </div>

        <div className="info-row">
          <span className="label">Member since:</span>
          <span className="value">{formatDate(state.user.created_at)}</span>
        </div>

        <div className="info-row">
          <span className="label">Status:</span>
          <span className={`status ${state.user.is_active ? 'active' : 'inactive'}`}>
            {state.user.is_active ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
