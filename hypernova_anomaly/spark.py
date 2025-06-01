from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType
import pandas as pd

# Yerel modülden Rust fonksiyonunu çağıran sarmalayıcıyı içe aktar
from .core import detect_evt_spot

def detect_spark(
    spark_df: DataFrame,
    column: str,
    q: float = 1e-5,
    level: float = 0.98
) -> DataFrame:
    """
    Bir Spark DataFrame üzerinde dağıtık EVT-SPOT anomali tespiti gerçekleştirir.

    Rapor II Bölüm V.A: Yüksek performanslı Pandas UDF'leri kullanır.
    SPOT sıralı bir algoritma olduğundan, her Spark bölümü (partition)
    bağımsız bir veri akışı olarak ele alınır. Bu, devasa paralellik sağlar.

    Args:
        spark_df: İşlem yapılacak Spark DataFrame.
        column: Anomali tespiti yapılacak sütunun adı.
        q: Risk parametresi.
        level: Başlangıç eşiği için kuantil seviyesi.

    Returns:
        DataFrame'e 'is_anomaly' sütunu eklenmiş yeni bir DataFrame.
    """

    @pandas_udf(BooleanType())
    def spot_udf(s: pd.Series) -> pd.Series:
        # Boş bölümleri atla
        if s.empty:
            return pd.Series([], dtype=bool)
            
        # Her bölüm için kalibrasyon boyutunu belirle
        n_init = int(len(s) * 0.2) # Bölümler daha küçük olabileceğinden oran artırıldı
        if n_init < 50:
             n_init = min(50, len(s) - 1)
        
        if n_init <= 0:
             return pd.Series([False] * len(s), dtype=bool)

        try:
            # Rust fonksiyonunu doğrudan PyArrow nesneleriyle çağır
            import pyarrow as pa
            arrow_array = pa.array(s, type=pa.float64())
            result_arrow = detect_evt_spot(arrow_array, q, n_init, level)
            return result_arrow.to_pandas()
        except Exception:
            # UDF içinde hata olursa, anomali olmadığını varsay
            return pd.Series([False] * len(s), dtype=bool)

    return spark_df.withColumn("is_anomaly", spot_udf(spark_df[column]))
