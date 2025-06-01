import pandas as pd
import pyarrow as pa
from typing import Union

# Derlenmiş Rust çekirdeğini içe aktar
from .core import detect_evt_spot

def detect(
    data: Union[pd.DataFrame, pd.Series],
    column: str = None,
    q: float = 1e-5,
    n_init_ratio: float = 0.1,
    level: float = 0.98
) -> pd.DataFrame:
    """
    EVT-SPOT algoritmasını kullanarak anomali tespiti yapar.
    Bu, Rapor II'de önerilen yenilikçi bir yaklaşımdır.

    Args:
        data: Anomali tespiti yapılacak olan Pandas DataFrame veya Series.
        column: DataFrame kullanılıyorsa işlem yapılacak sütunun adı.
        q: Anomali eşiği için risk parametresi (örn. 1e-4, 1e-5).
           Daha küçük değerler daha az anomali tespit eder.
        n_init_ratio: Verinin ne kadarının ilk kalibrasyon için kullanılacağı oranı.
        level: İlk 'peaks-over-threshold' (t) değerini belirlemek için
               kullanılan kuantil seviyesi.

    Returns:
        Orijinal veriye 'is_anomaly' bayrağını eklenmiş bir DataFrame.
    """
    if isinstance(data, pd.Series):
        series = data.astype(float)
        df_input = data.to_frame(name=data.name or "value")
    elif isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("DataFrame kullanılıyorsa 'column' parametresi zorunludur.")
        series = data[column].astype(float)
        df_input = data
    else:
        raise TypeError("Girdi verisi Pandas DataFrame veya Series olmalıdır.")

    n_init = int(len(series) * n_init_ratio)
    if n_init < 50:
        n_init = min(50, len(series) - 1)
        
    # 1. Pandas verisini PyArrow Float64Array'e dönüştür
    arrow_array = pa.array(series, type=pa.float64())

    # 2. Rust çekirdek fonksiyonunu çağır
    result_arrow_array = detect_evt_spot(arrow_array, q, n_init, level)

    # 3. Orijinal DataFrame'e sonuçları ekle
    output_df = df_input.copy()
    output_df['is_anomaly'] = result_arrow_array.to_pandas()

    return output_df
