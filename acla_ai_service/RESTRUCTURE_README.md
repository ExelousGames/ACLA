# ACLA AI Service - Restructured

This document explains the new folder structure for the ACLA AI Service.

## 📁 New Folder Structure

```
acla_ai_service/
├── app/                          # Main application package
│   ├── __init__.py              # Application initialization
│   ├── main.py                  # FastAPI application setup
│   ├── api/                     # API endpoints
│   │   ├── __init__.py         # API router aggregation
│   │   ├── health.py           # Health check endpoints
│   │   ├── datasets.py         # Dataset management endpoints
│   │   ├── ai.py               # AI-powered analysis endpoints
│   │   ├── racing_session.py   # Racing session analysis
│   │   ├── telemetry.py        # Telemetry analysis endpoints
│   │   ├── backend.py          # Backend integration endpoints
│   │   └── models.py           # ML model endpoints
│   ├── core/                   # Core configuration and utilities
│   │   ├── __init__.py         # Core module initialization
│   │   └── config.py           # Application configuration
│   ├── models/                 # Data models and schemas
│   │   ├── __init__.py         # Models initialization
│   │   ├── api_models.py       # API request/response models
│   │   └── telemetry_models.py # Telemetry data models
│   ├── services/               # Business logic services
│   │   ├── __init__.py         # Services initialization
│   │   ├── ai_service.py       # AI/OpenAI service
│   │   ├── telemetry_service.py # Telemetry processing service
│   │   ├── backend_service.py  # Backend integration service
│   │   └── analysis_service.py # Data analysis service
│   └── analyzers/              # Racing data analyzers
│       ├── __init__.py         # Analyzers initialization
│       └── advanced_analyzer.py # Advanced racing analysis
├── docs/                       # Documentation
│   ├── AI_INTELLIGENCE_GUIDE.md
│   ├── INTELLIGENCE_UPGRADE_SUMMARY.md
│   └── TELEMETRY_INTEGRATION.md
├── scripts/                    # Utility scripts and demos
│   ├── demo_intelligence_upgrade.py
│   └── telemetry_demo.py
├── tests/                      # Test files (to be created)
├── .env.example               # Environment variables example
├── requirements.txt           # Python dependencies
├── README.md                  # Main documentation
├── Dockerfile                 # Docker configuration
├── dev.dockerfile             # Development Docker configuration
└── main_new.py               # New entry point (temporary)
```

## 🔄 Migration from Old Structure

### Before (Flat Structure)
- All Python files were in the root directory
- No clear separation of concerns
- Difficult to navigate and maintain

### After (Modular Structure)
- Clear separation by functionality
- Easy to find and maintain code
- Better testing structure
- Scalable architecture

## 🚀 Key Improvements

### 1. **Separation of Concerns**
- **API Layer**: Clean endpoint definitions
- **Service Layer**: Business logic separated from API
- **Models Layer**: Data structures and validation
- **Core Layer**: Configuration and utilities

### 2. **Better Organization**
- **Documentation**: All docs in `/docs`
- **Scripts**: Demos and utilities in `/scripts`
- **Tests**: Dedicated test directory
- **Analyzers**: Racing-specific analysis logic

### 3. **Configuration Management**
- Centralized settings in `app/core/config.py`
- Environment-based configuration
- Type-safe settings with Pydantic

### 4. **API Structure**
- Modular routers for different functionalities
- Consistent response models
- Clear API versioning support

## 🔧 How to Use the New Structure

### Running the Application
```bash
# Using the new entry point
python main_new.py

# Or directly with uvicorn
uvicorn app.main:app --reload
```

### Adding New Features
1. **New API Endpoint**: Add to appropriate router in `/app/api/`
2. **New Business Logic**: Add service in `/app/services/`
3. **New Data Model**: Add to `/app/models/`
4. **New Analysis**: Add to `/app/analyzers/`

### Configuration
Update settings in `/app/core/config.py` or use environment variables.

## 📋 Next Steps

1. **Testing**: Create comprehensive tests in `/tests/`
2. **Migration**: Gradually migrate from old `main.py` to new structure
3. **Documentation**: Update API documentation
4. **CI/CD**: Update build scripts to use new structure

## 🔗 Dependencies

The new structure maintains all existing dependencies but organizes them better:
- FastAPI for API framework
- Pydantic for data validation
- OpenAI for AI features
- Scikit-learn for ML analysis
- Pandas/NumPy for data processing
- Plotly for visualizations

## 🐛 Known Issues

- Import paths need to be updated when fully migrating
- Some circular import dependencies may need resolution
- Testing framework needs to be set up

This restructure provides a solid foundation for scaling the ACLA AI Service while maintaining all existing functionality.
