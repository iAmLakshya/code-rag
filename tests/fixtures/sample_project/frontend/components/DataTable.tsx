/**
 * Advanced DataTable component with sorting, filtering, pagination, and virtualization.
 *
 * Demonstrates:
 * - Generic TypeScript components
 * - Render props pattern
 * - Compound components
 * - Custom hooks composition
 * - Memoization strategies
 * - Controlled vs uncontrolled modes
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  useRef,
  useEffect,
  ReactNode,
  ComponentType,
} from 'react';

// Type definitions

interface Column<T> {
  key: keyof T | string;
  header: string;
  width?: number | string;
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, row: T, index: number) => ReactNode;
}

type SortDirection = 'asc' | 'desc' | null;

interface SortState {
  column: string | null;
  direction: SortDirection;
}

interface FilterState {
  [key: string]: string;
}

interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

interface TableState<T> {
  data: T[];
  filteredData: T[];
  displayData: T[];
  sort: SortState;
  filters: FilterState;
  pagination: PaginationState;
  selectedRows: Set<string>;
  isLoading: boolean;
}

interface TableActions<T> {
  setSort: (column: string) => void;
  setFilter: (column: string, value: string) => void;
  clearFilters: () => void;
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
  selectRow: (id: string) => void;
  selectAll: () => void;
  clearSelection: () => void;
  refresh: () => Promise<void>;
}

interface DataTableProps<T extends { id: string }> {
  data?: T[];
  columns: Column<T>[];
  fetchData?: (params: {
    page: number;
    pageSize: number;
    sort: SortState;
    filters: FilterState;
  }) => Promise<{ data: T[]; total: number }>;
  defaultPageSize?: number;
  pageSizeOptions?: number[];
  selectable?: boolean;
  onSelectionChange?: (selectedIds: string[]) => void;
  rowKey?: keyof T;
  emptyMessage?: ReactNode;
  loadingComponent?: ReactNode;
  children?: (state: TableState<T>, actions: TableActions<T>) => ReactNode;
}

// Context for compound components

interface TableContextValue<T> {
  state: TableState<T>;
  actions: TableActions<T>;
  columns: Column<T>[];
}

const TableContext = createContext<TableContextValue<any> | null>(null);

function useTableContext<T>(): TableContextValue<T> {
  const context = useContext(TableContext);
  if (!context) {
    throw new Error('Table components must be used within DataTable');
  }
  return context as TableContextValue<T>;
}

// Custom hooks for table logic

function useSort<T>(
  data: T[],
  initialSort: SortState = { column: null, direction: null }
) {
  const [sort, setSort] = useState<SortState>(initialSort);

  const toggleSort = useCallback((column: string) => {
    setSort(current => {
      if (current.column !== column) {
        return { column, direction: 'asc' };
      }
      if (current.direction === 'asc') {
        return { column, direction: 'desc' };
      }
      return { column: null, direction: null };
    });
  }, []);

  const sortedData = useMemo(() => {
    if (!sort.column || !sort.direction) {
      return data;
    }

    return [...data].sort((a, b) => {
      const aVal = (a as any)[sort.column!];
      const bVal = (b as any)[sort.column!];

      if (aVal === bVal) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      const comparison = aVal < bVal ? -1 : 1;
      return sort.direction === 'asc' ? comparison : -comparison;
    });
  }, [data, sort]);

  return { sort, sortedData, toggleSort, setSort };
}

function useFilter<T>(data: T[], columns: Column<T>[]) {
  const [filters, setFilters] = useState<FilterState>({});

  const setFilter = useCallback((column: string, value: string) => {
    setFilters(current => {
      if (!value) {
        const { [column]: _, ...rest } = current;
        return rest;
      }
      return { ...current, [column]: value };
    });
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({});
  }, []);

  const filteredData = useMemo(() => {
    if (Object.keys(filters).length === 0) {
      return data;
    }

    return data.filter(row => {
      return Object.entries(filters).every(([column, filterValue]) => {
        const value = (row as any)[column];
        if (value === null || value === undefined) return false;
        return String(value).toLowerCase().includes(filterValue.toLowerCase());
      });
    });
  }, [data, filters]);

  return { filters, filteredData, setFilter, clearFilters };
}

function usePagination<T>(data: T[], defaultPageSize: number = 10) {
  const [pagination, setPagination] = useState<PaginationState>({
    page: 1,
    pageSize: defaultPageSize,
    total: data.length,
  });

  // Update total when data changes
  useEffect(() => {
    setPagination(current => ({
      ...current,
      total: data.length,
      page: Math.min(current.page, Math.ceil(data.length / current.pageSize) || 1),
    }));
  }, [data.length]);

  const setPage = useCallback((page: number) => {
    setPagination(current => ({ ...current, page }));
  }, []);

  const setPageSize = useCallback((pageSize: number) => {
    setPagination(current => ({
      ...current,
      pageSize,
      page: 1, // Reset to first page
    }));
  }, []);

  const paginatedData = useMemo(() => {
    const start = (pagination.page - 1) * pagination.pageSize;
    const end = start + pagination.pageSize;
    return data.slice(start, end);
  }, [data, pagination.page, pagination.pageSize]);

  const pageCount = Math.ceil(data.length / pagination.pageSize);

  return { pagination, paginatedData, setPage, setPageSize, pageCount };
}

function useSelection<T extends { id: string }>(data: T[]) {
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());

  const selectRow = useCallback((id: string) => {
    setSelectedRows(current => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedRows(new Set(data.map(row => row.id)));
  }, [data]);

  const clearSelection = useCallback(() => {
    setSelectedRows(new Set());
  }, []);

  const isAllSelected = useMemo(() => {
    return data.length > 0 && data.every(row => selectedRows.has(row.id));
  }, [data, selectedRows]);

  return { selectedRows, selectRow, selectAll, clearSelection, isAllSelected };
}

// Main DataTable component

function DataTable<T extends { id: string }>({
  data: propData,
  columns,
  fetchData,
  defaultPageSize = 10,
  pageSizeOptions = [10, 25, 50, 100],
  selectable = false,
  onSelectionChange,
  rowKey = 'id' as keyof T,
  emptyMessage = 'No data available',
  loadingComponent,
  children,
}: DataTableProps<T>) {
  const [localData, setLocalData] = useState<T[]>(propData ?? []);
  const [isLoading, setIsLoading] = useState(false);
  const isServerSide = !!fetchData;

  // Use local data or prop data
  const data = propData ?? localData;

  // Compose hooks
  const { sort, sortedData, toggleSort, setSort } = useSort(data);
  const { filters, filteredData, setFilter, clearFilters } = useFilter(
    isServerSide ? data : sortedData,
    columns
  );
  const { pagination, paginatedData, setPage, setPageSize } = usePagination(
    isServerSide ? data : filteredData,
    defaultPageSize
  );
  const { selectedRows, selectRow, selectAll, clearSelection } = useSelection(
    paginatedData
  );

  // Notify selection changes
  const prevSelection = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (
      onSelectionChange &&
      !setsAreEqual(selectedRows, prevSelection.current)
    ) {
      onSelectionChange(Array.from(selectedRows));
      prevSelection.current = new Set(selectedRows);
    }
  }, [selectedRows, onSelectionChange]);

  // Server-side data fetching
  const refresh = useCallback(async () => {
    if (!fetchData) return;

    setIsLoading(true);
    try {
      const result = await fetchData({
        page: pagination.page,
        pageSize: pagination.pageSize,
        sort,
        filters,
      });
      setLocalData(result.data);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setIsLoading(false);
    }
  }, [fetchData, pagination.page, pagination.pageSize, sort, filters]);

  // Fetch on mount and when params change (server-side only)
  useEffect(() => {
    if (isServerSide) {
      refresh();
    }
  }, [isServerSide, refresh]);

  // Build state and actions objects
  const state: TableState<T> = {
    data,
    filteredData: isServerSide ? data : filteredData,
    displayData: isServerSide ? data : paginatedData,
    sort,
    filters,
    pagination,
    selectedRows,
    isLoading,
  };

  const actions: TableActions<T> = {
    setSort: toggleSort,
    setFilter,
    clearFilters,
    setPage,
    setPageSize,
    selectRow,
    selectAll,
    clearSelection,
    refresh,
  };

  // Render prop pattern
  if (children) {
    return (
      <TableContext.Provider value={{ state, actions, columns }}>
        {children(state, actions)}
      </TableContext.Provider>
    );
  }

  // Default render
  return (
    <TableContext.Provider value={{ state, actions, columns }}>
      <div className="data-table">
        <TableFilters />
        <TableContent />
        <TablePagination pageSizeOptions={pageSizeOptions} />
      </div>
    </TableContext.Provider>
  );
}

// Compound components

function TableFilters() {
  const { state, actions, columns } = useTableContext();

  const filterableColumns = columns.filter(col => col.filterable !== false);

  if (filterableColumns.length === 0) return null;

  return (
    <div className="table-filters">
      {filterableColumns.map(column => (
        <input
          key={String(column.key)}
          type="text"
          placeholder={`Filter ${column.header}...`}
          value={state.filters[String(column.key)] || ''}
          onChange={e => actions.setFilter(String(column.key), e.target.value)}
        />
      ))}
      {Object.keys(state.filters).length > 0 && (
        <button onClick={actions.clearFilters}>Clear Filters</button>
      )}
    </div>
  );
}

function TableContent() {
  const { state, actions, columns } = useTableContext();

  if (state.isLoading) {
    return <div className="table-loading">Loading...</div>;
  }

  if (state.displayData.length === 0) {
    return <div className="table-empty">No data available</div>;
  }

  return (
    <table className="table">
      <TableHeader />
      <tbody>
        {state.displayData.map((row, index) => (
          <TableRow key={(row as any).id} row={row} index={index} />
        ))}
      </tbody>
    </table>
  );
}

function TableHeader() {
  const { state, actions, columns } = useTableContext();

  return (
    <thead>
      <tr>
        {columns.map(column => (
          <th
            key={String(column.key)}
            style={{ width: column.width }}
            onClick={() => column.sortable !== false && actions.setSort(String(column.key))}
            className={column.sortable !== false ? 'sortable' : ''}
          >
            {column.header}
            {state.sort.column === column.key && (
              <span className="sort-indicator">
                {state.sort.direction === 'asc' ? ' ↑' : ' ↓'}
              </span>
            )}
          </th>
        ))}
      </tr>
    </thead>
  );
}

function TableRow<T>({ row, index }: { row: T; index: number }) {
  const { columns } = useTableContext<T>();

  return (
    <tr>
      {columns.map(column => (
        <td key={String(column.key)}>
          {column.render
            ? column.render((row as any)[column.key], row, index)
            : String((row as any)[column.key] ?? '')}
        </td>
      ))}
    </tr>
  );
}

function TablePagination({ pageSizeOptions }: { pageSizeOptions: number[] }) {
  const { state, actions } = useTableContext();
  const { pagination } = state;

  const pageCount = Math.ceil(pagination.total / pagination.pageSize);

  return (
    <div className="table-pagination">
      <select
        value={pagination.pageSize}
        onChange={e => actions.setPageSize(Number(e.target.value))}
      >
        {pageSizeOptions.map(size => (
          <option key={size} value={size}>
            {size} per page
          </option>
        ))}
      </select>

      <div className="pagination-controls">
        <button
          onClick={() => actions.setPage(1)}
          disabled={pagination.page === 1}
        >
          First
        </button>
        <button
          onClick={() => actions.setPage(pagination.page - 1)}
          disabled={pagination.page === 1}
        >
          Previous
        </button>
        <span>
          Page {pagination.page} of {pageCount}
        </span>
        <button
          onClick={() => actions.setPage(pagination.page + 1)}
          disabled={pagination.page === pageCount}
        >
          Next
        </button>
        <button
          onClick={() => actions.setPage(pageCount)}
          disabled={pagination.page === pageCount}
        >
          Last
        </button>
      </div>

      <div className="pagination-info">
        Showing {(pagination.page - 1) * pagination.pageSize + 1} to{' '}
        {Math.min(pagination.page * pagination.pageSize, pagination.total)} of{' '}
        {pagination.total} entries
      </div>
    </div>
  );
}

// Higher-Order Component for adding table features

function withTableFeatures<P extends object>(
  WrappedComponent: ComponentType<P & { tableContext: TableContextValue<any> }>
) {
  return function WithTableFeatures(props: P) {
    const context = useTableContext();
    return <WrappedComponent {...props} tableContext={context} />;
  };
}

// Utility functions

function setsAreEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false;
  for (const item of a) {
    if (!b.has(item)) return false;
  }
  return true;
}

// Export everything
export {
  DataTable,
  TableFilters,
  TableContent,
  TableHeader,
  TableRow,
  TablePagination,
  useTableContext,
  withTableFeatures,
  useSort,
  useFilter,
  usePagination,
  useSelection,
  Column,
  SortState,
  FilterState,
  PaginationState,
  TableState,
  TableActions,
};
