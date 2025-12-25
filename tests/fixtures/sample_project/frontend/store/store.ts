/**
 * Global state management with Redux-like pattern.
 *
 * This module implements:
 * - Centralized state store
 * - Immutable state updates
 * - Action dispatching
 * - Middleware support (logging, async, devtools)
 * - Selector memoization
 * - State persistence
 *
 * Used throughout the application for:
 * - User authentication state
 * - Application settings
 * - Cached data
 * - UI state
 */

// Type definitions for the store system

type ActionType = string;

interface Action<T = any> {
  type: ActionType;
  payload?: T;
  meta?: {
    timestamp?: number;
    source?: string;
  };
  error?: boolean;
}

type Reducer<S, A extends Action = Action> = (state: S, action: A) => S;

type Middleware<S> = (store: Store<S>) => (
  next: Dispatch<Action>
) => (action: Action) => Action | Promise<Action>;

type Dispatch<A extends Action = Action> = (action: A) => A | Promise<A>;

type Subscriber<S> = (state: S, prevState: S) => void;

type Selector<S, R> = (state: S) => R;

interface StoreOptions<S> {
  initialState?: S;
  middleware?: Middleware<S>[];
  devtools?: boolean;
  persist?: {
    key: string;
    storage?: Storage;
    serialize?: (state: S) => string;
    deserialize?: (data: string) => S;
  };
}

/**
 * Creates a selector with memoization.
 * Only recomputes when input selector results change.
 */
function createSelector<S, R1, R>(
  selector1: Selector<S, R1>,
  combiner: (result1: R1) => R
): Selector<S, R>;
function createSelector<S, R1, R2, R>(
  selector1: Selector<S, R1>,
  selector2: Selector<S, R2>,
  combiner: (result1: R1, result2: R2) => R
): Selector<S, R>;
function createSelector<S, R1, R2, R3, R>(
  selector1: Selector<S, R1>,
  selector2: Selector<S, R2>,
  selector3: Selector<S, R3>,
  combiner: (result1: R1, result2: R2, result3: R3) => R
): Selector<S, R>;
function createSelector<S>(...args: any[]): Selector<S, any> {
  const combiner = args.pop();
  const selectors = args as Selector<S, any>[];

  let lastArgs: any[] | null = null;
  let lastResult: any = null;

  return (state: S) => {
    const currentArgs = selectors.map(s => s(state));

    // Check if any args changed
    if (lastArgs && currentArgs.every((arg, i) => arg === lastArgs![i])) {
      return lastResult;
    }

    lastArgs = currentArgs;
    lastResult = combiner(...currentArgs);
    return lastResult;
  };
}

/**
 * Main Store class implementing centralized state management.
 */
class Store<S extends object> {
  private state: S;
  private reducer: Reducer<S>;
  private subscribers: Set<Subscriber<S>> = new Set();
  private middleware: Middleware<S>[] = [];
  private isDispatching = false;

  constructor(reducer: Reducer<S>, options: StoreOptions<S> = {}) {
    this.reducer = reducer;
    this.state = options.initialState ?? ({} as S);
    this.middleware = options.middleware ?? [];

    // Load persisted state
    if (options.persist) {
      this.loadPersistedState(options.persist);
    }

    // Enable devtools
    if (options.devtools && typeof window !== 'undefined') {
      this.enableDevtools();
    }
  }

  /**
   * Get current state. Returns immutable reference.
   */
  getState(): Readonly<S> {
    if (this.isDispatching) {
      throw new Error('Cannot get state while dispatching');
    }
    return this.state;
  }

  /**
   * Dispatch an action to update state.
   * Actions flow through middleware before reaching reducer.
   */
  dispatch: Dispatch = (action: Action): Action | Promise<Action> => {
    if (this.isDispatching) {
      throw new Error('Cannot dispatch while dispatching');
    }

    // Add timestamp if not present
    if (!action.meta?.timestamp) {
      action = {
        ...action,
        meta: { ...action.meta, timestamp: Date.now() },
      };
    }

    // Build middleware chain
    const chain = this.middleware.map(mw => mw(this));
    let dispatch: Dispatch = this.baseDispatch.bind(this);

    for (let i = chain.length - 1; i >= 0; i--) {
      dispatch = chain[i](dispatch);
    }

    return dispatch(action);
  };

  /**
   * Base dispatch without middleware.
   */
  private baseDispatch(action: Action): Action {
    const prevState = this.state;

    try {
      this.isDispatching = true;
      this.state = this.reducer(this.state, action);
    } finally {
      this.isDispatching = false;
    }

    // Notify subscribers
    this.subscribers.forEach(subscriber => {
      try {
        subscriber(this.state, prevState);
      } catch (error) {
        console.error('Subscriber error:', error);
      }
    });

    return action;
  }

  /**
   * Subscribe to state changes.
   * Returns unsubscribe function.
   */
  subscribe(subscriber: Subscriber<S>): () => void {
    this.subscribers.add(subscriber);
    return () => {
      this.subscribers.delete(subscriber);
    };
  }

  /**
   * Replace the reducer (for hot reloading).
   */
  replaceReducer(nextReducer: Reducer<S>): void {
    this.reducer = nextReducer;
    this.dispatch({ type: '@@REPLACE_REDUCER' });
  }

  /**
   * Select derived data from state.
   */
  select<R>(selector: Selector<S, R>): R {
    return selector(this.getState());
  }

  /**
   * Load state from persistent storage.
   */
  private loadPersistedState(config: NonNullable<StoreOptions<S>['persist']>): void {
    const storage = config.storage ?? localStorage;
    const deserialize = config.deserialize ?? JSON.parse;

    try {
      const data = storage.getItem(config.key);
      if (data) {
        this.state = deserialize(data);
      }
    } catch (error) {
      console.warn('Failed to load persisted state:', error);
    }

    // Auto-persist on changes
    this.subscribe((state) => {
      const serialize = config.serialize ?? JSON.stringify;
      try {
        storage.setItem(config.key, serialize(state));
      } catch (error) {
        console.warn('Failed to persist state:', error);
      }
    });
  }

  /**
   * Connect to Redux DevTools extension.
   */
  private enableDevtools(): void {
    const devtools = (window as any).__REDUX_DEVTOOLS_EXTENSION__;
    if (devtools) {
      const connection = devtools.connect({ name: 'App Store' });
      connection.init(this.state);

      this.subscribe((state) => {
        connection.send({ type: 'STATE_UPDATE' }, state);
      });
    }
  }
}

// Built-in Middleware

/**
 * Logging middleware - logs all actions and state changes.
 */
const loggerMiddleware: Middleware<any> = (store) => (next) => (action) => {
  console.group(`Action: ${action.type}`);
  console.log('Payload:', action.payload);
  console.log('Prev State:', store.getState());

  const result = next(action);

  console.log('Next State:', store.getState());
  console.groupEnd();

  return result;
};

/**
 * Thunk middleware - allows dispatching functions for async actions.
 */
const thunkMiddleware: Middleware<any> = (store) => (next) => (action) => {
  if (typeof action === 'function') {
    return action(store.dispatch, store.getState);
  }
  return next(action);
};

/**
 * Promise middleware - handles promise payloads.
 */
const promiseMiddleware: Middleware<any> = (store) => (next) => async (action) => {
  if (action.payload instanceof Promise) {
    try {
      const payload = await action.payload;
      return next({ ...action, payload });
    } catch (error) {
      return next({ ...action, payload: error, error: true });
    }
  }
  return next(action);
};

// Combine reducers utility

function combineReducers<S extends object>(
  reducers: { [K in keyof S]: Reducer<S[K]> }
): Reducer<S> {
  const keys = Object.keys(reducers) as (keyof S)[];

  return (state = {} as S, action: Action): S => {
    let hasChanged = false;
    const nextState = {} as S;

    for (const key of keys) {
      const reducer = reducers[key];
      const prevStateForKey = state[key];
      const nextStateForKey = reducer(prevStateForKey, action);

      nextState[key] = nextStateForKey;
      hasChanged = hasChanged || nextStateForKey !== prevStateForKey;
    }

    return hasChanged ? nextState : state;
  };
}

// Action creators utility

type ActionCreator<P = void> = P extends void
  ? () => Action<void>
  : (payload: P) => Action<P>;

function createAction<P = void>(type: string): ActionCreator<P> {
  const actionCreator = (payload?: P): Action<P> => ({
    type,
    payload: payload as P,
    meta: { timestamp: Date.now() },
  });

  actionCreator.type = type;
  actionCreator.match = (action: Action): action is Action<P> =>
    action.type === type;

  return actionCreator as ActionCreator<P>;
}

// Export everything
export {
  Store,
  Action,
  Reducer,
  Middleware,
  Dispatch,
  Subscriber,
  Selector,
  StoreOptions,
  createSelector,
  combineReducers,
  createAction,
  loggerMiddleware,
  thunkMiddleware,
  promiseMiddleware,
};

// Application-specific state types

interface User {
  id: string;
  email: string;
  username: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

interface UIState {
  theme: 'light' | 'dark';
  sidebarOpen: boolean;
  notifications: Array<{
    id: string;
    message: string;
    type: 'info' | 'success' | 'error';
  }>;
}

interface AppState {
  auth: AuthState;
  ui: UIState;
}

// Auth actions
const loginRequest = createAction<{ email: string; password: string }>('auth/loginRequest');
const loginSuccess = createAction<{ user: User; token: string }>('auth/loginSuccess');
const loginFailure = createAction<string>('auth/loginFailure');
const logout = createAction('auth/logout');

// UI actions
const toggleTheme = createAction('ui/toggleTheme');
const toggleSidebar = createAction('ui/toggleSidebar');
const addNotification = createAction<{ message: string; type: 'info' | 'success' | 'error' }>('ui/addNotification');
const removeNotification = createAction<string>('ui/removeNotification');

// Auth reducer
const authReducer: Reducer<AuthState> = (
  state = { user: null, token: null, isAuthenticated: false, isLoading: false, error: null },
  action
) => {
  if (loginRequest.match(action)) {
    return { ...state, isLoading: true, error: null };
  }
  if (loginSuccess.match(action)) {
    return {
      ...state,
      user: action.payload.user,
      token: action.payload.token,
      isAuthenticated: true,
      isLoading: false,
    };
  }
  if (loginFailure.match(action)) {
    return { ...state, error: action.payload, isLoading: false };
  }
  if (logout.match(action)) {
    return { user: null, token: null, isAuthenticated: false, isLoading: false, error: null };
  }
  return state;
};

// UI reducer
const uiReducer: Reducer<UIState> = (
  state = { theme: 'light', sidebarOpen: true, notifications: [] },
  action
) => {
  if (toggleTheme.match(action)) {
    return { ...state, theme: state.theme === 'light' ? 'dark' : 'light' };
  }
  if (toggleSidebar.match(action)) {
    return { ...state, sidebarOpen: !state.sidebarOpen };
  }
  if (addNotification.match(action)) {
    return {
      ...state,
      notifications: [
        ...state.notifications,
        { id: Date.now().toString(), ...action.payload },
      ],
    };
  }
  if (removeNotification.match(action)) {
    return {
      ...state,
      notifications: state.notifications.filter(n => n.id !== action.payload),
    };
  }
  return state;
};

// Combined root reducer
const rootReducer = combineReducers<AppState>({
  auth: authReducer,
  ui: uiReducer,
});

// Selectors
const selectAuth = (state: AppState) => state.auth;
const selectUI = (state: AppState) => state.ui;
const selectUser = createSelector(selectAuth, auth => auth.user);
const selectIsAuthenticated = createSelector(selectAuth, auth => auth.isAuthenticated);
const selectTheme = createSelector(selectUI, ui => ui.theme);
const selectNotifications = createSelector(selectUI, ui => ui.notifications);

// Create store instance
const store = new Store(rootReducer, {
  middleware: [thunkMiddleware, promiseMiddleware, loggerMiddleware],
  devtools: process.env.NODE_ENV === 'development',
  persist: {
    key: 'app-state',
    storage: typeof localStorage !== 'undefined' ? localStorage : undefined,
  },
});

export {
  store,
  AppState,
  AuthState,
  UIState,
  loginRequest,
  loginSuccess,
  loginFailure,
  logout,
  toggleTheme,
  toggleSidebar,
  addNotification,
  removeNotification,
  selectUser,
  selectIsAuthenticated,
  selectTheme,
  selectNotifications,
};
