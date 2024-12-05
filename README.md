# NCA Analysis Tool

A Streamlit-based tool for Necessary Condition Analysis (NCA)

## Features

- Interactive data preprocessing with visualizations
- Multiple NCA methods (CE-FDH, CR-FDH, Quantile)
- Statistical analysis with bootstrapping
- Dynamic result visualization
- Export capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gerandi/nca-analysis-tool.git
cd nca-analysis-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main application entry point
- `src/`: Core application code
  - `core/`: Analysis and data processing logic
  - `ui/`: User interface components
  - `utils/`: Utility functions and constants
- `tests/`: Test cases

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.