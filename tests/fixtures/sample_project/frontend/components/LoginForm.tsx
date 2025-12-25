/**
 * Login form component for user authentication.
 */

import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';

interface LoginFormProps {
  onSuccess?: () => void;
  onError?: (error: string) => void;
}

interface FormState {
  email: string;
  password: string;
  isLoading: boolean;
  error: string | null;
}

/**
 * LoginForm component handles user authentication.
 *
 * Features:
 * - Email and password input
 * - Form validation
 * - Loading state during submission
 * - Error handling and display
 */
export const LoginForm: React.FC<LoginFormProps> = ({ onSuccess, onError }) => {
  const { login } = useAuth();
  const [state, setState] = useState<FormState>({
    email: '',
    password: '',
    isLoading: false,
    error: null,
  });

  /**
   * Validate form inputs before submission.
   */
  const validateForm = (): boolean => {
    if (!state.email || !state.email.includes('@')) {
      setState(prev => ({ ...prev, error: 'Please enter a valid email' }));
      return false;
    }
    if (!state.password || state.password.length < 8) {
      setState(prev => ({ ...prev, error: 'Password must be at least 8 characters' }));
      return false;
    }
    return true;
  };

  /**
   * Handle form submission.
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await login(state.email, state.password);
      onSuccess?.();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Login failed';
      setState(prev => ({ ...prev, error: errorMessage }));
      onError?.(errorMessage);
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  /**
   * Handle input field changes.
   */
  const handleInputChange = (field: 'email' | 'password') => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setState(prev => ({
      ...prev,
      [field]: e.target.value,
      error: null, // Clear error on input change
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="login-form">
      <h2>Login</h2>

      {state.error && (
        <div className="error-message" role="alert">
          {state.error}
        </div>
      )}

      <div className="form-group">
        <label htmlFor="email">Email</label>
        <input
          type="email"
          id="email"
          value={state.email}
          onChange={handleInputChange('email')}
          disabled={state.isLoading}
          placeholder="Enter your email"
          autoComplete="email"
        />
      </div>

      <div className="form-group">
        <label htmlFor="password">Password</label>
        <input
          type="password"
          id="password"
          value={state.password}
          onChange={handleInputChange('password')}
          disabled={state.isLoading}
          placeholder="Enter your password"
          autoComplete="current-password"
        />
      </div>

      <button
        type="submit"
        disabled={state.isLoading}
        className="submit-button"
      >
        {state.isLoading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};

export default LoginForm;
