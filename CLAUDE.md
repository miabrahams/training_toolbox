# Training Toolbox - User Interface Structure

This repository is a collection of multiple tools for machine learning training workflows, particularly focused on diffusion models and image processing. The codebase is organized into several key directories that support different aspects of the training pipeline.

## Repository Structure

The Training Toolbox contains multiple specialized tools organized in the following directories:

### Core UI Components (`src/ui/`)
- **Main UI Framework**: Uses Gradio for web-based interface
- **Tab-based Architecture**: Each major feature is implemented as a separate tab
- **Modular Design**: Individual tab components can be developed and maintained independently

### Analysis Libraries (`lib/`)
- **ComfyUI Integration**: Tools for analyzing ComfyUI workflow metadata
- **Prompt Processing**: Utilities for extracting and analyzing generation prompts
- **Media Processing**: FFmpeg integration, image utilities, and metadata handling

### Data Processing (`src/`)
- **Tag Analysis**: Advanced prompt and tag analysis tools with clustering capabilities
- **Duplicate Detection**: Multiple algorithms for finding duplicate images
- **Batch Processing**: Tools for processing large datasets

### Specialized Tools
- **Captioning System** (`captioner/`): Text caption processing and collation
- **ComfyUI Automation** (`ComfyAutomation/`): Automated workflow processing
- **Web Scraping** (`scrape/`): Data collection from various sources
- **Training Scripts** (`src/training/`): Model training utilities

## Current UI Tabs

The main interface (`ui.py`) provides access to several specialized tools through a tabbed interface:

### 1. ComfyUI Prompt Extractor
- **Purpose**: Extract positive/negative prompts from ComfyUI-generated images
- **Features**: 
  - Drag & drop image upload
  - Automatic metadata parsing
  - Dual prompt extraction (positive/negative)
  - Copy-to-clipboard functionality
- **Dependencies**: None (works standalone)

### 2. Frame Extractor
- **Purpose**: Extract frames from video files for training datasets
- **Features**:
  - Video file selection and processing
  - Configurable frame extraction rates
  - WSL path handling for cross-platform compatibility
- **Location**: `src/ui/frame_extractor_tab.py`

### 3. Tag Analysis
- **Purpose**: Advanced analysis of prompt data with clustering
- **Features**:
  - Semantic clustering of prompts
  - Tag frequency analysis
  - Interactive cluster exploration
  - Statistical summaries
- **Dependencies**: Requires database initialization
- **Location**: `src/ui/tag_analysis_tab.py`

### 4. Prompt Search
- **Purpose**: Search and explore prompt datasets
- **Features**:
  - Full-text search capabilities
  - Filter by various criteria
  - Batch operations
- **Dependencies**: Requires analyzer state
- **Location**: `src/ui/prompt_search_tab.py`

### 5. Direct Search
- **Purpose**: Direct database queries for prompt data
- **Features**:
  - Raw database access
  - Custom query capabilities
  - Export functionality
- **Dependencies**: Requires database state
- **Location**: `src/ui/direct_search_tab.py`

## Usage Patterns

### Standalone Tools
Some tabs (like ComfyUI Prompt Extractor) work independently and can be used immediately without any setup.

### Database-Dependent Tools
Other tabs require initialization of the database and analyzer components:
1. Load database path and data directory
2. Initialize prompt data and analyzer
3. Access advanced analysis features

### Development Workflow
- Each tab is implemented as a separate Python module in `src/ui/`
- The main `ui.py` file orchestrates tab creation and shared state management
- Shared utilities are centralized in `lib/` for reuse across tools

## Key Design Principles

1. **Modularity**: Each tool can be developed and maintained independently
2. **Progressive Enhancement**: Basic features work without setup, advanced features require initialization
3. **Shared State**: Database and analyzer state is shared across relevant tabs
4. **Error Handling**: Graceful degradation when dependencies are not available
5. **Responsive Design**: Clean, functional interface using Gradio components

## Future Extensibility

The architecture supports easy addition of new tabs by:
1. Creating a new module in `src/ui/`
2. Implementing the tab creation function
3. Importing and registering in `ui.py`
4. Optionally connecting to shared state for database/analyzer access

This structure allows the Training Toolbox to continue growing as a comprehensive suite of ML training utilities while maintaining clean separation of concerns.