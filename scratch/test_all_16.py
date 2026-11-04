import lxml.etree as ET
from latex2mathml.converter import convert as latex2mathml_convert

xslt_path = r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'
xslt = ET.parse(xslt_path)
transform = ET.XSLT(xslt)

equations = [
    r"\min_{j, s} \left[ \sum_{i: x_{ij} \le s} (y_i - \bar{y}_{R_1})^2 + \sum_{i: x_{ij} > s} (y_i - \bar{y}_{R_2})^2 \right]",
    r"\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2",
    r"F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})",
    r"\ln(1 + \text{revenue}_{s,t}) = \mu_s + \delta_{s,t}",
    r"\hat{\mu}_s = \frac{n_s}{n_s + m} \hat{\mu}_s^{\text{history}} + \frac{m}{n_s + m} \hat{\mu}_s^{\text{prior}} \quad (m = 10)",
    r"\hat{y}_{\log}^{\text{Ensemble}} = \sum_{k=1}^K w_k \hat{y}_{\log}^{(k)} \quad (w_k \ge 0, \sum_{k=1}^K w_k = 1)",
    r"\min_{\mathbf{w}} \text{RMSLE}\left(\mathbf{y}_{\text{val}}, \exp\left( \sum_{k=1}^K w_k \hat{y}_{\log, \text{val}}^{(k)} \right) - 1 \right)",
    r"\text{RMSLE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \ln(1 + \max(0, \hat{y}_i)) - \ln(1 + y_i) \right)^2}",
    r"\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2_2 + \alpha \|\mathbf{w}\|^2_2",
    r"\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2_2 + \lambda \|\mathbf{w}\|_1",
    r"\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)",
    r"\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2",
    r"\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|",
    r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}",
    r"R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}",
    r"C = (F - 32) \times \frac{5}{9}"
]

for i, eq in enumerate(equations):
    try:
        mathml = latex2mathml_convert(eq)
        tree = ET.fromstring(mathml)
        omml = transform(tree)
        root = omml.getroot()
        print(f"Eq {i+1}: SUCCESS")
    except Exception as e:
        print(f"Eq {i+1}: ERROR -> {e}")
