# Hypernova Anomaly: Yüksek Hızda Univariate Anomali Tespit Motoru

Bu kütüphane, normal dağılıma uymayan büyük veri kümelerinde anomali tespiti için dünyanın en hızlısı olma hedefiyle geliştirilmiş, son teknoloji bir motordur. Çekirdeği, bellek güvenliği ve ham performansı birleştiren **Rust** dilinde yazılmıştır.

## Felsefe ve Algoritma

Bu motor, geleneksel yöntemlerin ötesine geçerek, **Aşırı Değer Teorisi (EVT)** temelinde çalışan **SPOT (Streaming Peaks-Over-Threshold)** algoritmasını kullanır. Bu yaklaşım:

-   **İstatistiksel Olarak Sağlamdır**: Verinin tamamını modellemek yerine, tanım gereği anomalilerin bulunduğu kuyrukları doğrudan modeller.
-   **Dinamik ve Uyarlanabilirdir**: Gelen veriye göre anomali eşiğini otomatik olarak ayarlar.
-   **Hiper-Optimize Edilmiştir**: Çekirdek algoritma, hız için **Momentler Metodu (MOM)** gibi hesaplama açısından ucuz ve etkili teknikler kullanır.

## Kurulum

### Gereksinimler
- Python 3.8+
- Rust & Cargo (https://rustup.rs/)

### Kurulum Adımları
1.  **Sanal Ortam Oluşturun:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  **Geliştirme Modunda Kurulum:**
    Rust kodunu derlemek ve Python paketini kurmak için `maturin` kullanın.
    ```bash
    pip install maturin
    maturin develop
    ```
    Bu komut sonrası kütüphane kullanıma hazırdır.

## Kullanım

### Pandas ile Hızlı Başlangıç

```python
import pandas as pd
import numpy as np
from hypernova_anomaly import detect

# 1000 normal nokta ve 5 belirgin aykırı değer
data = np.concatenate([np.random.randn(1000) * 5 + 20, [100, 110, -40, -55, 150]])
df = pd.DataFrame(data, columns=['sensor_reading'])

# Anomali tespiti
result = detect(df, column='sensor_reading', q=1e-6, level=0.99)

print("Tespit edilen anomaliler:")
print(result[result['is_anomaly']])
```
### Apache Spark ile Milyarlarca Satırda Çalışma
Örnek kullanım için `examples/full_usage_example.py` dosyasına bakınız.
```bash
# Gerekli kütüphaneleri kurun
pip install pyspark pandas

# Örneği çalıştırın
spark-submit examples/full_usage_example.py
```
#### `examples/full_usage_example.py`
Hem Pandas hem de Spark kullanımını gösteren detaylı bir örnek.

```python
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

def run_pandas_example():
    print("-" * 50)
    print("PANDAS KULLANIM ÖRNEĞİ BAŞLATILIYOR")
    print("-" * 50)
    
    from hypernova_anomaly import detect

    # Veri: 1000 normal nokta ve 5 belirgin aykırı değer
    data = np.concatenate([
        np.random.randn(1000) * 5 + 20, 
        [100, 110, -40, -55, 150]
    ])
    np.random.shuffle(data)
    df = pd.DataFrame(data, columns=['sensor_reading'])
    
    print(f"Toplam veri noktası: {len(df)}")

    # Anomali tespiti
    # q: Çok küçük bir risk seviyesi. Daha küçük q, daha bariz anomalileri bulur.
    # level: Verinin %99'unun normal olduğunu varsayarak başlangıç eşiğini belirler.
    result_df = detect(df, column='sensor_reading', q=1e-6, level=0.99)

    anomalies = result_df[result_df['is_anomaly']]
    print(f"\nTespit edilen anomali sayısı: {len(anomalies)}")
    print("Tespit edilen anomaliler:")
    print(anomalies)
    print("-" * 50)


def run_spark_example():
    print("\n" + "-" * 50)
    print("APACHE SPARK KULLANIM ÖRNEĞİ BAŞLATILIYOR")
    print("-" * 50)

    spark = SparkSession.builder \
        .appName("Hypernova Anomaly Spark Example") \
        .master("local[4]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
        
    from hypernova_anomaly.spark import detect_spark

    # 1 milyon normal nokta oluştur
    n_rows = 1_000_000
    df = spark.range(n_rows).withColumn("value", (np.random.randn() * 10 + 50))

    # Birkaç aşırı aykırı değer ekle
    outliers = spark.createDataFrame([(1000.0,), (-500.0,), (2500.0,)], ["value"])
    df_with_anomalies = df.union(outliers).repartition(4) # Paralelliği sağlamak için

    print(f"Toplam veri noktası: {df_with_anomalies.count()}")
    
    # Dağıtık anomali tespiti
    result_df = detect_spark(df_with_anomalies, column="value", q=1e-7, level=0.995)

    print("\nAnomali tespiti tamamlandı. Anormal olarak işaretlenenler:")
    result_df.filter("is_anomaly = true").show()

    spark.stop()
    print("-" * 50)

if __name__ == "__main__":
    run_pandas_example()
    run_spark_example()
```
