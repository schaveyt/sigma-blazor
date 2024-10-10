# RAG Solution for C# Web Framework Documentation

This project implements a Retrieval-Augmented Generation (RAG) solution for querying documentation of a C# web framework. It uses a folder of markdown files as its knowledge base and leverages Anthropic's Claude model for generating responses.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:

   ```sh
   git clone https://github.com/yourusername/rag-csharp-webframework.git
   cd rag-csharp-webframework
   ```

2. Create a virtual environment (optional but recommended):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

4. Set up your Anthropic API key:
   - Sign up for an account at https://www.anthropic.com
   - Obtain your API key
   - Add your API key to the configuration file (see Customization section)

### Usage

1. Place your markdown files in a directory, for example, `docs/`.

2. Update the `docs_path` in the configuration file to point to your markdown files directory.

3. Run the script:

   ```sh
   python rag_solution.py
   ```

4. Enter your questions when prompted. The system will provide answers based on the documentation.

5. Type 'quit' to exit the program.

## Customization

- Create a `config.json` file in the project root with the following structure:
  
  ```json
  {
    "model_name": "microsoft/codebert-base",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "anthropic_model": "claude-2",
    "anthropic_api_key": "your_api_key_here"
  }
  ```

- Adjust these parameters as needed for your specific use case.

## Hardware Requirements

To run this RAG solution effectively, the following hardware specifications are recommended:

1. **CPU**: A modern multi-core processor (e.g., Intel i7 or AMD Ryzen 7) with at least 4 cores.
   - Why: Enables efficient parallel processing of documents and handling of multiple queries.

1. **RAM**: At least 16GB of RAM, with 32GB or more recommended.
   - Why: RAG solutions can be memory-intensive, especially with large documentation sets. Sufficient RAM ensures embedding models and vector stores can be held in memory.

1. **Storage**: SSD storage with at least 100GB free space.
   - Why: SSDs provide faster read/write speeds, crucial for quick document loading and vector store operations.

1. **GPU** (Optional but recommended): NVIDIA GTX 1660 or better, with at least 6GB of VRAM.
   - Why: Can significantly speed up embedding generation and similarity searches.

1. **Internet Connection**: Stable and fast internet connection.
   - Why: Required for reliable communication with the Anthropic API.

Note: These are general recommendations. Actual requirements may vary based on the size of your documentation and expected query load. For production environments, consider using a cloud platform that can provide scalable resources.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.