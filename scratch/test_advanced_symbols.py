import lxml.etree as ET
from latex2mathml.converter import convert as latex2mathml_convert

xslt_path = r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'
xslt = ET.parse(xslt_path)
transform = ET.XSLT(xslt)

tests = [
    r"f: \mathcal{X} \to \mathcal{Y}",
    r"\mathbf{x} \in \mathbb{R}^d",
    r"y \in \mathbb{R}",
    r"D = \{(x_{s,t}, y_{s,t})\}",
    r"s \in \{1, 2, \dots, S\}",
    r"t \in \{1, 2, \dots, T\}",
    r"(\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}",
    r"z = \frac{x - \mu}{\sigma}",
    r"\Delta_{\text{RMSLE}} = \text{Val RMSLE} - \text{Train RMSLE} = 0.1218 - 0.1033 = 0.0185",
    r"\beta = 0.000318",
    r"R^2 = 0.9055",
    r"\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2",
    r"\min_{\mathbf{w}} \text{RMSLE}\left(\mathbf{y}_{\text{val}}, \exp\left( \sum_{k=1}^K w_k \hat{y}_{\log, \text{val}}^{(k)} \right) - 1 \right) \quad \text{s.t.} \quad w_k \ge 0, \sum w_k = 1"
]

for i, t in enumerate(tests):
    try:
        mathml = latex2mathml_convert(t)
        tree = ET.fromstring(mathml)
        omml = transform(tree)
        print(f"Test {i+1}: SUCCESS")
    except Exception as e:
        print(f"Test {i+1}: ERROR -> {e}")
