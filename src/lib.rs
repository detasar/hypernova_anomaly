use arrow_rs::array::{ArrayRef, BooleanArray, Float64Array, PrimitiveArray};
use arrow_rs::datatypes::Float64Type;
use arrow_rs::record_batch::RecordBatch;
use ndarray::prelude::*;
use ndarray_stats::{QuantileExt,interpolate::Linear};
use pyo3::prelude::*;
use pyo3_arrow::PyArrowType;
use rayon::prelude::*;
use thiserror::Error;

// Özel hata türleri
#[derive(Error, Debug)]
pub enum SpotError {
    #[error("İlk kalibrasyon verisi (n_init) yetersiz: {0} noktadan az olamaz.")]
    InsufficientInitialData(usize),
    #[error("Başlangıç eşiği (level) için %99.9'dan yüksek bir değer seçilemez.")]
    LevelTooHigh,
    #[error("Hesaplama sırasında varyans sıfır çıktı, GPD uydurulamıyor.")]
    ZeroVarianceError,
}

// SPOT algoritmasının durumunu tutan ana yapı
struct Spot {
    q: f64,             // Risk parametresi (örn. 1e-4)
    t: f64,             // Mevcut yüksek eşik (peaks-over-threshold)
    num_total: usize,   // İşlenen toplam nokta sayısı
    num_peaks: usize,   // Eşiği aşan nokta sayısı
    peaks: Vec<f64>,    // Eşiği aşan değerler (aşım miktarları)
    gamma: f64,         // GPD şekil parametresi
    sigma: f64,         // GPD ölçek parametresi
}

impl Spot {
    /// Rapor Bölüm VI.A'daki sözde koda göre SPOT algoritmasını başlatır.
    fn new(q: f64, initial_data: ArrayView1<f64>, level: f64) -> Result<Self, SpotError> {
        let n_init = initial_data.len();
        if n_init < 30 { // Makul bir başlangıç için minimum boyut
            return Err(SpotError::InsufficientInitialData(30));
        }
        if level > 0.999 {
            return Err(SpotError::LevelTooHigh);
        }

        let mut initial_data_mut = initial_data.to_owned();
        let t = *initial_data_mut.quantile_mut(level, &Linear).unwrap();

        let peaks: Vec<f64> = initial_data.iter().filter(|&&x| x > t).map(|&x| x - t).collect();
        let num_peaks = peaks.len();
        if num_peaks == 0 {
            // Hiç zirve yoksa, varsayılan parametrelerle başla
            return Ok(Spot { q, t, num_total: n_init, num_peaks: 0, peaks: vec![], gamma: 0.1, sigma: 1.0 });
        }
        
        // Rapor Bölüm III.B'de önerilen hızlı GPD uydurma: Momentler Metodu (MOM)
        let (gamma, sigma) = Self::fit_gpd_mom(&peaks)?;

        Ok(Spot { q, t, num_total: n_init, num_peaks, peaks, gamma, sigma })
    }

    /// Momentler Metodu (MOM) ile GPD parametrelerini hesaplar.
    /// Hız için MLE veya LME'ye tercih edilmiştir.
    fn fit_gpd_mom(peaks: &[f64]) -> Result<(f64, f64), SpotError> {
        let n = peaks.len() as f64;
        if n < 2.0 {
             return Ok((0.1, 1.0)); // Çok az zirve varsa varsayılan
        }
        let peak_mean = peaks.iter().sum::<f64>() / n;
        let peak_var = peaks.iter().map(|&p| (p - peak_mean).powi(2)).sum::<f64>() / n;

        if peak_var.abs() < 1e-9 {
            return Err(SpotError::ZeroVarianceError);
        }

        let gamma = 0.5 * ( (peak_mean.powi(2) / peak_var) - 1.0 );
        let sigma = peak_mean * (0.5 + 0.5 * (peak_mean.powi(2) / peak_var));
        Ok((gamma, sigma))
    }

    /// Anomali eşiği olan z_q'yu hesaplar.
    fn calculate_zq(&self) -> f64 {
        let nt = self.num_peaks as f64;
        let n = self.num_total as f64;
        
        if self.gamma.abs() < 1e-9 { // gamma ~ 0 durumu (Üstel Dağılım)
            return self.t - self.sigma * (self.q * n / nt).ln();
        }
        
        let term = (self.q * n / nt).powf(-self.gamma);
        self.t + (self.sigma / self.gamma) * (term - 1.0)
    }

    /// Veri akışındaki tek bir noktayı işler ve anomali olup olmadığını döndürür.
    fn process_point(&mut self, x: f64) -> bool {
        self.num_total += 1;
        let zq = self.calculate_zq();

        if x > zq {
            // Anomali tespit edildi
            true
        } else if x > self.t {
            // Anomali değil ama "gerçek zirve". Modeli güncelle.
            self.num_peaks += 1;
            self.peaks.push(x - self.t);
            if let Ok((gamma, sigma)) = Self::fit_gpd_mom(&self.peaks) {
                self.gamma = gamma;
                self.sigma = sigma;
            }
            false
        } else {
            // Normal nokta
            false
        }
    }
}


/// Python'dan çağrılacak ana, yüksek performanslı fonksiyon.
/// Rapor Bölüm V: Apache Arrow üzerinden verimli veri aktarımı.
#[pyfunction]
fn detect_evt_spot(
    data_array: PyArrowType<Float64Array>,
    q: f64,
    n_init: usize,
    level: f64,
) -> PyResult<PyArrowType<BooleanArray>> {
    let data_ref: &Float64Array = &data_array;
    let data = data_ref.values();

    if data.len() <= n_init {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Veri boyutu, başlangıç boyutu (n_init) kadar veya daha az olamaz."
        ));
    }

    // Adım 1: Rapor Bölüm VI.A - Algoritmayı kalibre et
    let initial_data = ArrayView1::from(&data[..n_init]);
    let mut spot_model = Spot::new(q, initial_data, level)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Adım 2: Kalan verileri işle
    // Not: Buradaki `map` paralel değildir çünkü SPOT durumu sıralı olarak günceller.
    // Paralellik, Spark katmanında veri bölümlerini ayırarak sağlanır.
    let mut flags: Vec<bool> = vec![false; data.len()];
    for i in n_init..data.len() {
        flags[i] = spot_model.process_point(data[i]);
    }

    let result_array = BooleanArray::from(flags);
    Ok(PyArrowType(result_array))
}

// Python modülünü oluştur
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_evt_spot, m)?)?;
    Ok(())
}
