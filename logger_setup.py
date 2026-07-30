import logging
import uuid
from pythonjsonlogger import jsonlogger
from datetime import datetime

class ContextualJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(ContextualJsonFormatter, self).add_fields(log_record, record, message_dict)
        # Inject standard telemetry markers into every log line
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Ensure a correlation ID always exists for distributed tracing
        if not log_record.get('correlation_id'):
            log_record['correlation_id'] = f"TRACE-{uuid.uuid4().hex[:8].upper()}"

def setup_production_logging(logger_name: str = "legal_nlp_platform") -> logging.Logger:
    """Configures and returns a highly performant structured JSON logger instance."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if re-initialized
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Define the structural schema fields to extract
        formatter = ContextualJsonFormatter('%(timestamp)s %(level)s %(logger)s %(message)s %(correlation_id)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Example usage check
if __name__ == "__main__":
    log = setup_production_logging()
    log.info("System monitoring initialized successfully.", extra={"correlation_id": "TRACE-PROD-12345"})