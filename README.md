# neutron-stars

Herramientas en **Python** para experimentar con **estrellas de neutrones**: manejo/visualización de **Ecuaciones de Estado (EoS)** y scripts de **estructura estelar relativista** (ecuaciones **TOV**) en unidades CGS y *geometrizadas*.

> ⚠️ Nota: el repo está orientado a scripts (no como paquete instalable). Si algún script falla por rutas/imports, mira la sección **“Ejecución de scripts”**.

---

## Contenido

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución de scripts](#ejecución-de-scripts)
  - [Plot de EoS](#plot-de-eos)
  - [TOV (densidad constante) + solución analítica](#tov-densidad-constante--solución-analítica)
- [Formato de archivos EoS](#formato-de-archivos-eos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Notas sobre unidades](#notas-sobre-unidades)
- [Contribuir](#contribuir)
- [Licencia](#licencia)
- [Créditos](#créditos)

---

## Características

- 📈 **Visualización de EoS** (curvas `log10(p)` vs `log10(ρ)` y un zoom de la región densa).
- 🧰 Utilidades para **extraer columnas** de densidad y presión desde un `.txt` de EoS.
- 📚 Módulos con **constantes físicas** y **conversiones** entre CGS y unidades geometrizadas.
- 🌌 Funciones para **ecuaciones TOV** y **solución interior de Schwarzschild** (densidad uniforme).
- 🧪 Script demostración: integración/plots para el caso de **ρ constante** y comparación con solución analítica.

---

## Requisitos

- Python 3.x
- Dependencias:
  - `numpy`
  - `matplotlib`

> Si más adelante añades un `requirements.txt`, esta sección se puede simplificar.

---

## Instalación

Clona el repositorio y crea un entorno virtual:

```bash
git clone https://github.com/Chenoveko/neutron-stars.git
cd neutron-stars

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install numpy matplotlib
