/**
 * API utilities and hooks for data fetching.
 */

import { useState, useEffect, useCallback } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

interface ApiError {
  status: number;
  message: string;
}

interface ApiResponse<T> {
  data: T;
  status: number;
}

/**
 * Get the current auth token from localStorage.
 */
function getAuthToken(): string | null {
  return localStorage.getItem('auth_token');
}

/**
 * Make an authenticated API request.
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getAuthToken();

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...options.headers,
  };

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw {
      status: response.status,
      message: error.error || 'Request failed',
    } as ApiError;
  }

  const result = await response.json();
  return result.data;
}

/**
 * Fetch the current user's profile.
 */
export async function fetchUserProfile() {
  return apiRequest('/auth/profile');
}

/**
 * Fetch a list of items with pagination.
 */
export async function fetchItems(page: number = 1, limit: number = 10) {
  return apiRequest(`/items?page=${page}&limit=${limit}`);
}

/**
 * Create a new item.
 */
export async function createItem(data: Record<string, unknown>) {
  return apiRequest('/items', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * Update an existing item.
 */
export async function updateItem(id: number, data: Record<string, unknown>) {
  return apiRequest(`/items/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

/**
 * Delete an item.
 */
export async function deleteItem(id: number) {
  return apiRequest(`/items/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Custom hook for data fetching with loading and error states.
 */
export function useApi<T>(
  fetchFn: () => Promise<T>,
  dependencies: unknown[] = []
) {
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const execute = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await fetchFn();
      setData(result);
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setIsLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    execute();
  }, [execute]);

  return {
    data,
    isLoading,
    error,
    refetch: execute,
  };
}

export default useApi;
