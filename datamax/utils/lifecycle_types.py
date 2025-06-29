from enum import Enum


class LifeType(Enum):
    # 数据处理阶段
    DATA_PROCESSING = "DATA_PROCESSING"  # 正在处理
    DATA_PROCESSED = "DATA_PROCESSED"  # 处理完成
    DATA_PROCESS_FAILED = "DATA_PROCESS_FAILED"  # 处理失败

    # 数据清洗阶段
    DATA_CLEANING = "DATA_CLEANING"  # 正在清洗
    DATA_CLEANED = "DATA_CLEANED"  # 清洗完成
    DATA_CLEAN_FAILED = "DATA_CLEAN_FAILED"  # 清洗失败

    # 数据标注阶段
    DATA_LABELLING = "DATA_LABELLING"  # 正在标注
    DATA_LABELLED = "DATA_LABELLED"  # 标注完成
    DATA_LABEL_FAILED = "DATA_LABEL_FAILED"  # 标注失败
