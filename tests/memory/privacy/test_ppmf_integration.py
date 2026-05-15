"""Integration tests for PPMF with Memory class."""
import pytest


class TestPPMFIntegration:
    """Test PPMF integration with Memory class."""
    
    @pytest.mark.skip(reason="Requires OpenAI API key for full Memory instantiation")
    def test_ppmf_initialization(self):
        """Test that PPMF initializes correctly with Memory."""
        from outhad_contextkit import Memory
        from outhad_contextkit.configs.base import MemoryConfig
        from outhad_contextkit.memory.privacy.config import PPMFConfig
        
        config = MemoryConfig(ppmf=PPMFConfig(enabled=True))
        memory = Memory(config=config)
        
        assert memory.ppmf_enabled
        assert hasattr(memory, '_memory_sanitizer')
        assert hasattr(memory, '_encryption_manager')
        assert hasattr(memory, '_telemetry_sanitizer')
    
    def test_ppmf_config_included(self):
        """Test that PPMFConfig is properly included in MemoryConfig."""
        from outhad_contextkit.configs.base import MemoryConfig
        from outhad_contextkit.memory.privacy.config import PPMFConfig
        
        config = MemoryConfig()
        assert hasattr(config, 'ppmf')
        assert isinstance(config.ppmf, PPMFConfig)
        assert config.ppmf.enabled == False  # Default is disabled
    
    def test_ppmf_config_override(self):
        """Test that PPMF config can be overridden."""
        from outhad_contextkit.configs.base import MemoryConfig
        from outhad_contextkit.memory.privacy.config import PPMFConfig
        from outhad_contextkit.memory.privacy.enums import EncryptionAlgorithm, RedactionMode
        
        ppmf_config = PPMFConfig(
            enabled=True,
            use_llm_classifier=False,
            encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
            default_redaction_mode=RedactionMode.MASK,
            redact_telemetry=True
        )
        config = MemoryConfig(ppmf=ppmf_config)
        
        assert config.ppmf.enabled == True
        assert config.ppmf.use_llm_classifier == False
        assert config.ppmf.encryption_algorithm == EncryptionAlgorithm.AES_256_GCM
        assert config.ppmf.default_redaction_mode == RedactionMode.MASK
        assert config.ppmf.redact_telemetry == True
    
    def test_ppmf_components_importable(self):
        """Test that all PPMF components can be imported."""
        from outhad_contextkit.memory.privacy import (
            EncryptionManager,
            MemorySanitizer,
            PPMFConfig,
            PrivacyLevel,
            RedactionMode,
            SensitiveSpanType,
            TelemetrySanitizer,
        )
        
        # Just verify they're all importable
        assert EncryptionManager is not None
        assert MemorySanitizer is not None
        assert TelemetrySanitizer is not None
        assert PPMFConfig is not None
        assert PrivacyLevel is not None
        assert RedactionMode is not None
        assert SensitiveSpanType is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

