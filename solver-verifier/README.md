# IMO 2025 Problem Solver

A parallel AI agent system for solving International Mathematical Olympiad (IMO) problems using Google's Gemini API.

```
MIT License

Copyright (c) 2025 Lin Yang, Yichen Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Overview

entire process :

![1755094793472](image/README/1755094793472.png)

This project consists of two main components:

- `agent.py`: A single AI agent that attempts to solve IMO problems
- `run_parallel.py`: A parallel execution system that runs multiple agents simultaneously

## Prerequisites

1. **Python 3.7+** installed on your system
2. **Google Gemini API key** - Get one from [Google AI Studio](https://aistudio.google.com/app/apikey)
3. **Required Python packages**:
   ```bash
   pip install requests
   ```

## Setup

1. **Clone or download the project files**
2. **Set up your API key**:
   - Create a `.env` file in the project directory
   - Add your API key: `GOOGLE_API_KEY=your_api_key_here`
   - Or set it as an environment variable: `export GOOGLE_API_KEY=your_api_key_here`

## Usage

### Single Agent (`agent.py`)

Run a single agent to solve an IMO problem:

```bash
python agent.py problem.txt [options]
```

**Arguments:**

- `problem.txt`: Path to the problem statement file (required); imo2025 problems are in `problems`

**Options:**

- `--log LOG_FILE`: Specify a log file for output (default: prints to console)
- `--other_prompts PROMPTS`: Additional prompts separated by commas

**Example:**

```bash
python agent.py imo2025_p1.txt --log agent_output.log
```

### Parallel Execution (`run_parallel.py`)

Run multiple agents in parallel to increase the chance of finding a solution:

```bash
python run_parallel.py problem.txt [options]
```

**Arguments:**

- `problem.txt`: Path to the problem statement file (required)

**Options:**

- `--num-agents N` or `-n N`: Number of parallel agents (default: 10)
- `--log-dir DIR` or `-d DIR`: Directory for log files (default: logs)
- `--timeout SECONDS` or `-t SECONDS`: Timeout per agent in seconds (default: no timeout)
- `--max-workers N` or `-w N`: Maximum worker processes (default: number of agents)
- `--other_prompts PROMPTS` or `-o PROMPTS`: Additional prompts separated by commas

**Examples:**

```bash
# Run 20 agents with 5-minute timeout each
python run_parallel.py imo2025_p1.txt -n 20 -t 300

# Run 5 agents with custom log directory
python run_parallel.py imo2025_p1.txt -n 5 -d my_logs

# Run with additional prompts
python run_parallel.py imo2025_p1.txt -n 15 -o "focus_on_geometry,use_induction"
```

## Problem File Format

See the `problems` folder.

## Output and Logging

### Single Agent

- Output is printed to console by default
- Use `--log` to save output to a file
- The agent will indicate if a complete solution was found

### Parallel Execution

- Each agent creates a separate log file in the specified directory
- Progress is shown in real-time
- Final summary shows:
  - Total execution time
  - Number of successful/failed agents
  - Success rate
  - Which agent found a solution (if any)
  - Location of log files

## Understanding the Output

### Solution Detection

The system looks for the phrase "Found a correct solution in run" to identify successful solutions.

### Agent Behavior

- Agents use Google's Gemini 2.5 Pro model
- Each agent follows a structured approach with multiple attempts
- Solutions are verified for completeness and correctness
- Agents can provide partial solutions if complete solutions aren't found

## Tips for Best Results

1. **Problem Formatting**: Ensure your problem file is clear and well-formatted
2. **Parallel Execution**: Use more agents for harder problems (10-20 agents recommended)
3. **Timeout Settings**: Set reasonable timeouts (you may set no timeout)
4. **API Limits**: Be aware of Google API rate limits and costs
5. **Log Analysis**: Check individual agent logs for detailed reasoning

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Google API key is properly set
2. **Timeout Issues**: Increase timeout or reduce number of agents
3. **Memory Issues**: Reduce max-workers if running out of memory
4. **No Solutions Found**: Try running more agents or check problem clarity

### Debug Mode

Add verbose logging by modifying the agent code or check individual log files for detailed output.

## License

MIT License - Copyright (c) 2025 Lin Yang, Yichen Huang

This software is provided as-is. Users are free to copy, modify, and distribute the code with proper attribution.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

### Community Codes

Community contributions are located in `code/community_codes/`. These have not been thoroughly tested, so please use them at your own risk.

## Disclaimer

This tool is for educational and research purposes. Success in solving IMO problems depends on problem difficulty and AI model capabilities. Not all problems may be solvable by the current system.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{huang2025gemini,
  title={Gemini 2.5 Pro Capable of Winning Gold at IMO 2025},
  author={Huang, Yichen and Yang, Lin F},
  journal={arXiv preprint arXiv:2507.15855},
  year={2025}
}
```
