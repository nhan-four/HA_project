"""
Logger - Hệ thống logging cho module Hedge Algebra Clustering

Cung cấp:
- Logging ra console với màu sắc
- Logging ra file với timestamp
- Các mức log: DEBUG, INFO, WARNING, ERROR
"""

import logging
import os
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter với màu sắc cho console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format: [TIME] [LEVEL] message
        record.levelname = f"{color}{record.levelname:8}{reset}"
        return super().format(record)


def get_logger(
    name: str = "HedgeAlgebra",
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Tạo và trả về logger với cấu hình đã định.
    
    Args:
        name: Tên của logger
        level: Mức độ logging (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Có ghi log ra file không
        log_dir: Thư mục chứa file log
    
    Returns:
        logging.Logger: Logger đã được cấu hình
    
    Example:
        >>> logger = get_logger("MyModule", level="DEBUG")
        >>> logger.info("Bắt đầu training...")
        >>> logger.debug("Chi tiết: theta=0.5, alpha=0.5")
    """
    logger = logging.getLogger(name)
    
    # Tránh thêm handler trùng lặp
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler với màu sắc
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = ColoredFormatter(
        fmt='[%(asctime)s] %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"hedge_algebra_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log tất cả vào file
        file_format = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.debug(f"Log file: {log_file}")
    
    return logger


class TrainingLogger:
    """
    Logger chuyên dụng cho quá trình training.
    Ghi lại các thông tin quan trọng theo dạng có cấu trúc.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.training_history = []
    
    def log_config(self, config: dict):
        """Ghi log cấu hình."""
        self.logger.info("=" * 60)
        self.logger.info("CẤU HÌNH TRAINING")
        self.logger.info("=" * 60)
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
    
    def log_data_info(self, n_samples: int, n_features: int, n_classes: int):
        """Ghi log thông tin dữ liệu."""
        self.logger.info(f"📊 Dữ liệu:")
        self.logger.info(f"  • Số samples: {n_samples:,}")
        self.logger.info(f"  • Số features: {n_features:,}")
        self.logger.info(f"  • Số classes: {n_classes:,}")
    
    def log_clustering_iteration(self, iteration: int, centers: list, converged: bool):
        """Ghi log mỗi vòng lặp clustering."""
        centers_str = ", ".join([f"{c:.4f}" for c in centers])
        status = "✓ Hội tụ" if converged else ""
        self.logger.debug(f"  Iteration {iteration:3d}: centers=[{centers_str}] {status}")
    
    def log_cluster_distribution(self, cluster_counts: dict):
        """Ghi log phân bố các cụm."""
        self.logger.info(f"📊 Phân bố cụm:")
        for cluster_id, count in cluster_counts.items():
            self.logger.info(f"  • Cụm {cluster_id}: {count:,} samples")
    
    def log_training_result(self, cluster_id: int, success: bool, error: str = None):
        """Ghi log kết quả training của từng cụm."""
        if success:
            self.logger.info(f"  ✅ Cụm {cluster_id}: Training thành công")
        else:
            self.logger.error(f"  ❌ Cụm {cluster_id}: Training thất bại - {error}")
    
    def log_metrics(self, metrics: dict):
        """Ghi log các metrics đánh giá."""
        self.logger.info("=" * 60)
        self.logger.info("📈 KẾT QUẢ ĐÁNH GIÁ")
        self.logger.info("=" * 60)
        for name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {name}: {value:.4f}")
            else:
                self.logger.info(f"  {name}: {value}")
        self.logger.info("=" * 60)
    
    def log_summary(self, train_time: float, test_time: float, accuracy: float):
        """Ghi log tóm tắt cuối cùng."""
        self.logger.info("=" * 60)
        self.logger.info("📋 TÓM TẮT")
        self.logger.info("=" * 60)
        self.logger.info(f"  ⏱️  Thời gian training: {train_time:.2f}s")
        self.logger.info(f"  ⏱️  Thời gian testing: {test_time:.4f}s")
        self.logger.info(f"  🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info("=" * 60)

