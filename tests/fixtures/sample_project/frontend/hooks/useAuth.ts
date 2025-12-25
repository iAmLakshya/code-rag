/**
 * Authentication hook for managing user sessions.
 */

import { useState, useCallback, useEffect, createContext, useContext } from 'react';

interface User {
  id: number;
  email: string;
  username: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (email: string, username: string, password: string) => Promise<void>;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

/**
 * Custom hook for authentication functionality.
 *
 * Provides:
 * - Login/logout/register methods
 * - Current user state
 * - Token management
 * - Persistence via localStorage
 */
export function useAuth(): AuthContextValue {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: null,
    isAuthenticated: false,
    isLoading: true,
  });

  // Initialize from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    const storedUser = localStorage.getItem('auth_user');

    if (storedToken && storedUser) {
      setState({
        user: JSON.parse(storedUser),
        token: storedToken,
        isAuthenticated: true,
        isLoading: false,
      });
    } else {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  /**
   * Authenticate user with email and password.
   */
  const login = useCallback(async (email: string, password: string): Promise<void> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Login failed');
      }

      const data = await response.json();
      const { token, user } = data.data;

      // Persist to localStorage
      localStorage.setItem('auth_token', token);
      localStorage.setItem('auth_user', JSON.stringify(user));

      setState({
        user,
        token,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false }));
      throw error;
    }
  }, []);

  /**
   * Log out the current user.
   */
  const logout = useCallback(async (): Promise<void> => {
    const { token } = state;

    if (token) {
      try {
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
      } catch (error) {
        console.warn('Logout request failed:', error);
      }
    }

    // Clear localStorage
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');

    setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  }, [state.token]);

  /**
   * Register a new user account.
   */
  const register = useCallback(async (
    email: string,
    username: string,
    password: string
  ): Promise<void> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, username, password }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Registration failed');
      }

      // Auto-login after registration
      await login(email, password);
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false }));
      throw error;
    }
  }, [login]);

  return {
    ...state,
    login,
    logout,
    register,
  };
}

export default useAuth;
