"""
Batch Processor - Xử lý dữ liệu theo batch để tránh tràn RAM

Hỗ trợ:
- Xử lý dataset lớn (hàng triệu dòng)
- Giới hạn RAM sử dụng
- Generator-based processing
- Memory monitoring
"""

import numpy as np
import psutil
import gc
from typing import Generator, Tuple, Optional, Callable, Iterator
from dataclasses import dataclass

from .logger import get_logger


@dataclass
class MemoryConfig:
    """
    Cấu hình giới hạn bộ nhớ.
    
    Attributes:
        max_memory_gb: RAM tối đa được sử dụng (GB)
        batch_size: Số samples mỗi batch (tự động tính nếu None)
        reserve_memory_gb: RAM dự trữ cho hệ thống (GB)
    """
    max_memory_gb: float = 4.0
    batch_size: Optional[int] = None
    reserve_memory_gb: float = 1.0
    
    def __post_init__(self):
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb phải > 0")
        if self.reserve_memory_gb < 0:
            raise ValueError("reserve_memory_gb phải >= 0")


class MemoryManager:
    """
    Quản lý bộ nhớ RAM.
    
    Theo dõi và kiểm soát việc sử dụng RAM để tránh tràn bộ nhớ.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger("MemoryManager", level="WARNING", log_to_file=False)
        
        # Tính toán giới hạn thực tế
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.max_bytes = min(
            config.max_memory_gb * (1024**3),
            (total_ram_gb - config.reserve_memory_gb) * (1024**3)
        )
        
        self.logger.info(f"Memory limit: {self.max_bytes / (1024**3):.2f} GB")
    
    def get_current_usage(self) -> float:
        """Trả về RAM đang sử dụng (bytes)."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_available_memory(self) -> float:
        """Trả về RAM còn khả dụng (bytes)."""
        return self.max_bytes - self.get_current_usage()
    
    def estimate_array_size(self, shape: Tuple[int, ...], dtype=np.float64) -> int:
        """Ước tính kích thước array (bytes)."""
        return np.prod(shape) * np.dtype(dtype).itemsize
    
    def calculate_optimal_batch_size(
        self,
        n_features: int,
        dtype=np.float64,
        safety_factor: float = 0.7
    ) -> int:
        """
        Tính batch size tối ưu dựa trên RAM khả dụng.
        
        Args:
            n_features: Số features
            dtype: Kiểu dữ liệu
            safety_factor: Hệ số an toàn (0-1)
        
        Returns:
            int: Batch size tối ưu
        """
        if self.config.batch_size is not None:
            return self.config.batch_size
        
        available = self.get_available_memory() * safety_factor
        bytes_per_sample = n_features * np.dtype(dtype).itemsize
        
        # Cần thêm bộ nhớ cho các biến trung gian (~3x)
        batch_size = int(available / (bytes_per_sample * 3))
        
        # Giới hạn trong khoảng hợp lý
        batch_size = max(100, min(batch_size, 100000))
        
        return batch_size
    
    def force_garbage_collection(self):
        """Buộc thu gom rác để giải phóng RAM."""
        gc.collect()
    
    def check_memory_ok(self, required_bytes: int) -> bool:
        """Kiểm tra có đủ RAM không."""
        return self.get_available_memory() >= required_bytes


class BatchProcessor:
    """
    Xử lý dữ liệu theo batch để tránh tràn RAM.
    
    Cho phép xử lý dataset cực lớn (hàng triệu dòng) mà không
    cần load toàn bộ vào RAM.
    
    Example:
        >>> processor = BatchProcessor(max_memory_gb=4.0)
        >>> for batch_X, batch_y, batch_info in processor.iterate_batches(X, y):
        ...     # Xử lý batch
        ...     process(batch_X, batch_y)
    """
    
    def __init__(
        self,
        max_memory_gb: float = 4.0,
        batch_size: Optional[int] = None,
        log_level: str = "INFO"
    ):
        """
        Khởi tạo BatchProcessor.
        
        Args:
            max_memory_gb: RAM tối đa được sử dụng (GB)
            batch_size: Số samples mỗi batch (None = tự động)
            log_level: Mức độ logging
        """
        self.memory_config = MemoryConfig(
            max_memory_gb=max_memory_gb,
            batch_size=batch_size
        )
        self.memory_manager = MemoryManager(self.memory_config)
        self.logger = get_logger("BatchProcessor", level=log_level, log_to_file=False)
    
    def iterate_batches(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], dict], None, None]:
        """
        Iterate qua dữ liệu theo batch.
        
        Args:
            X: Features array
            y: Labels array (optional)
            batch_size: Kích thước batch (None = tự động)
        
        Yields:
            Tuple[batch_X, batch_y, batch_info]
            - batch_X: Features của batch
            - batch_y: Labels của batch (None nếu không có y)
            - batch_info: Dict chứa thông tin batch (start_idx, end_idx, batch_num)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Tính batch size
        if batch_size is None:
            batch_size = self.memory_manager.calculate_optimal_batch_size(n_features)
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        self.logger.info(f"📦 Batch processing: {n_samples:,} samples, {n_batches} batches, size={batch_size:,}")
        
        for batch_num in range(n_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx] if y is not None else None
            
            batch_info = {
                'batch_num': batch_num,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_samples': end_idx - start_idx,
                'total_batches': n_batches
            }
            
            yield batch_X, batch_y, batch_info
            
            # Garbage collection sau mỗi batch
            if batch_num % 10 == 0:
                self.memory_manager.force_garbage_collection()
    
    def process_in_batches(
        self,
        X: np.ndarray,
        process_func: Callable[[np.ndarray], np.ndarray],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Xử lý dữ liệu theo batch và ghép kết quả.
        
        Args:
            X: Input array
            process_func: Hàm xử lý mỗi batch
            batch_size: Kích thước batch
        
        Returns:
            np.ndarray: Kết quả đã ghép
        """
        results = []
        
        for batch_X, _, batch_info in self.iterate_batches(X, batch_size=batch_size):
            batch_result = process_func(batch_X)
            results.append(batch_result)
            
            self.logger.debug(
                f"   Batch {batch_info['batch_num']+1}/{batch_info['total_batches']}"
            )
        
        return np.concatenate(results, axis=0)
    
    def calculate_semantic_values_batched(
        self,
        X: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Tính semantic values theo batch.
        
        Args:
            X: Features array
            batch_size: Kích thước batch
        
        Returns:
            np.ndarray: Semantic values
        """
        return self.process_in_batches(
            X,
            lambda batch: np.mean(batch, axis=1),
            batch_size=batch_size
        )


class ChunkedFileReader:
    """
    Đọc file lớn theo chunks để tránh tràn RAM.
    
    Hỗ trợ CSV và NPY files.
    """
    
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 10000,
        max_memory_gb: float = 4.0
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.logger = get_logger("ChunkedReader", level="INFO", log_to_file=False)
    
    def read_csv_chunks(
        self,
        label_column: Optional[str] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Đọc CSV theo chunks.
        
        Args:
            label_column: Tên cột label
        
        Yields:
            Tuple[X_chunk, y_chunk]
        """
        import pandas as pd
        
        self.logger.info(f"📂 Đọc file theo chunks: {self.file_path}")
        
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            if label_column is not None:
                y = chunk[label_column].values
                X = chunk.drop(columns=[label_column]).values
            else:
                y = chunk.iloc[:, -1].values
                X = chunk.iloc[:, :-1].values
            
            yield X.astype(np.float64), y
    
    def count_rows(self) -> int:
        """Đếm số dòng trong file (không load vào RAM)."""
        import subprocess
        result = subprocess.run(['wc', '-l', self.file_path], capture_output=True, text=True)
        return int(result.stdout.split()[0]) - 1  # Trừ header


def estimate_memory_requirement(
    n_samples: int,
    n_features: int,
    n_clusters: int,
    dtype=np.float64
) -> float:
    """
    Ước tính RAM cần thiết (GB).
    
    Args:
        n_samples: Số samples
        n_features: Số features
        n_clusters: Số cụm
        dtype: Kiểu dữ liệu
    
    Returns:
        float: RAM cần thiết (GB)
    """
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Bộ nhớ cho dữ liệu
    data_memory = n_samples * n_features * bytes_per_element
    
    # Bộ nhớ cho labels và semantic values
    labels_memory = n_samples * bytes_per_element * 2
    
    # Bộ nhớ cho các biến trung gian
    intermediate_memory = data_memory * 0.5
    
    total_bytes = data_memory + labels_memory + intermediate_memory
    
    return total_bytes / (1024**3)

