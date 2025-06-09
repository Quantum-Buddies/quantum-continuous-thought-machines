# Quantum Continuous Thought Machine (Quantum CTM) Architecture

## Overview

The Quantum CTM is a hybrid quantum-classical neural network that implements the principles of Continuous Thought Machines using quantum memory cells and quantum synchronization. The architecture allows for iterative refinement of thoughts through a recurrent processing loop.

## Architecture Diagram

```mermaid
graph TB
    %% Input Layer
    Input["Input Data<br/>(batch_size, 784)<br/>MNIST Images"] --> InputLayer["Input Layer<br/>Linear(784 → 128)"]
    
    %% Initial Processing
    InputLayer --> ProcessedInput["Processed Input<br/>(batch_size, 128)"]
    ProcessedInput --> KV["Key-Value Store<br/>(batch_size, 1, 128)"]
    ProcessedInput --> InitState["Initial Hidden State<br/>(batch_size, 128)"]
    
    %% State Trace Initialization
    InitState --> StateTrace["State Trace<br/>(batch_size, 128, 8)<br/>History Buffer"]
    
    %% Recurrent Thought Loop
    StateTrace --> RecurrentLoop{{"Recurrent Thought Loop<br/>(iterations = 5)"}}
    
    %% Inside the Loop
    RecurrentLoop --> FlatTrace["Flatten Trace<br/>(batch_size, 1024)"]
    
    %% Quantum Memory Processing
    FlatTrace --> QMemoryBank["Quantum Memory Bank<br/>(3 slots)"]
    
    subgraph QMemoryBank["Quantum Memory Bank"]
        direction TB
        
        %% Slot 0
        TraceProc0["Trace Processor 0<br/>MLP: 1024→256→4"]
        QNN0["Quantum Memory Cell 0<br/>2 qubits, 4 params"]
        OutputMap0["Output Mapper 0<br/>Linear: 4→128"]
        
        %% Slot 1
        TraceProc1["Trace Processor 1<br/>MLP: 1024→256→4"]
        QNN1["Quantum Memory Cell 1<br/>2 qubits, 4 params"]
        OutputMap1["Output Mapper 1<br/>Linear: 4→128"]
        
        %% Slot 2
        TraceProc2["Trace Processor 2<br/>MLP: 1024→256→4"]
        QNN2["Quantum Memory Cell 2<br/>2 qubits, 4 params"]
        OutputMap2["Output Mapper 2<br/>Linear: 4→128"]
    end
    
    %% Memory Cell Connections
    FlatTrace --> TraceProc0 --> QNN0 --> OutputMap0
    FlatTrace --> TraceProc1 --> QNN1 --> OutputMap1
    FlatTrace --> TraceProc2 --> QNN2 --> OutputMap2
    
    %% Memory Output
    OutputMap0 --> MemoryOutput["Memory Output<br/>(batch_size, 128)"]
    OutputMap1 --> MemoryOutput
    OutputMap2 --> MemoryOutput
    
    %% State Update
    MemoryOutput --> Combine1["Concatenate<br/>[hidden, memory]<br/>(batch_size, 256)"]
    InitState --> Combine1
    Combine1 --> StateUpdater["State Updater<br/>Linear(256→128) + ReLU"]
    StateUpdater --> UpdatedState["Updated Hidden State<br/>(batch_size, 128)"]
    
    %% Update State Trace
    UpdatedState --> UpdateTrace["Update State Trace<br/>Shift left + append"]
    UpdateTrace --> StateTrace
    
    %% Quantum Synchronization
    StateTrace --> QSync["Quantum Synchronization Layer"]
    
    subgraph QSync["Quantum Synchronization Layer"]
        direction LR
        Corr01["Correlation Circuit 0-1<br/>4 qubits total"]
        Corr02["Correlation Circuit 0-2<br/>4 qubits total"]
        Corr12["Correlation Circuit 1-2<br/>4 qubits total"]
    end
    
    %% Synchronization Output
    QSync --> SyncVector["Sync Vector<br/>(batch_size, 3)"]
    SyncVector --> SyncSplit{{"Split Sync Vector"}}
    SyncSplit --> SyncAction["Sync Action<br/>(batch_size, 1)"]
    SyncSplit --> SyncOut["Sync Out<br/>(batch_size, 2)"]
    
    %% Attention Mechanism
    SyncAction --> QProj["Query Projection<br/>Linear(1→128)"]
    QProj --> Query["Query<br/>(batch_size, 1, 128)"]
    Query --> Attention["Multi-Head Attention<br/>(4 heads)"]
    KV --> Attention
    Attention --> AttnOut["Attention Output<br/>(batch_size, 128)"]
    
    %% Final State Update
    AttnOut --> Combine2["Concatenate<br/>[hidden, attention]"]
    UpdatedState --> Combine2
    Combine2 --> StateUpdater2["State Updater<br/>Linear(256→128) + ReLU"]
    StateUpdater2 --> FinalState["Final Hidden State<br/>(batch_size, 128)"]
    
    %% Output Projection
    SyncOut --> OutputProj["Output Projector<br/>Linear(2→10)"]
    OutputProj --> StepOutput["Step Output<br/>(batch_size, 10)"]
    
    %% Loop Control
    StepOutput --> CheckIter{{"More Iterations?"}}
    CheckIter -->|Yes| RecurrentLoop
    CheckIter -->|No| FinalOutput["Final Output<br/>(batch_size, 10)<br/>MNIST Classes"]
    
    %% Style
    classDef quantum fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    classDef classical fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef loop fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    
    class QNN0,QNN1,QNN2,Corr01,Corr02,Corr12 quantum
    class InputLayer,TraceProc0,TraceProc1,TraceProc2,OutputMap0,OutputMap1,OutputMap2,StateUpdater,StateUpdater2,QProj,Attention,OutputProj classical
    class Input,ProcessedInput,KV,InitState,StateTrace,FlatTrace,MemoryOutput,UpdatedState,SyncVector,SyncAction,SyncOut,Query,AttnOut,FinalState,StepOutput,FinalOutput data
    class RecurrentLoop,CheckIter loop
```

## Component Descriptions

### Classical Components

1. **Input Layer**: Linear transformation from input dimension (784 for MNIST) to hidden dimension (128)
2. **Trace Processors**: MLPs that compress the state history (1024D) down to quantum-compatible parameters (4D)
3. **Output Mappers**: Linear layers that project quantum measurement probabilities back to hidden dimension
4. **State Updater**: ReLU-activated linear layer that updates the hidden state based on memory and attention outputs
5. **Attention Mechanism**: Multi-head attention (4 heads) that uses synchronization output to focus on input features
6. **Output Projector**: Final linear layer that maps synchronization output to class predictions

### Quantum Components

1. **Quantum Memory Cells**: 
   - 3 independent slots, each with 2 qubits
   - Parameterized by RY and RZ rotation gates
   - Stores compressed representations of the state history

2. **Quantum Synchronization Layer**:
   - Creates correlation circuits between memory slot pairs (0-1, 0-2, 1-2)
   - Measures entanglement/correlation between quantum states
   - Produces synchronization metrics for controlling attention and output

### Data Flow

1. **Input Processing**: MNIST images are flattened and transformed to hidden dimension
2. **Recurrent Loop**: The model iterates 5 times, refining its internal state
3. **Memory Access**: State history is compressed and stored in quantum memory
4. **Synchronization**: Quantum correlations guide attention and output generation
5. **Output Generation**: Final predictions are based on the synchronization vector

### Key Features

- **Iterative Refinement**: Multiple thought steps allow complex reasoning
- **Quantum Memory**: Exploits quantum superposition for efficient state storage
- **Quantum Correlation**: Uses entanglement to measure relationships between memory slots
- **Hybrid Processing**: Combines classical neural networks with quantum circuits
- **Attention Control**: Quantum synchronization drives classical attention mechanism

## Implementation Notes

- Built with PyTorch and Qiskit
- Uses Qiskit's `TorchConnector` for automatic differentiation through quantum circuits
- Supports batch processing for efficient training
- Modular design allows easy experimentation with different quantum architectures 