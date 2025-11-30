-- Create dedicated table for tracking file processing status
CREATE TABLE IF NOT EXISTS sdc_ai.qa_file_status (
    id BIGSERIAL PRIMARY KEY,
    source_key VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING', -- PENDING, PROCESSING, COMPLETED, FAILED
    run_id BIGINT, -- Reference to the execution run
    error_message TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
    
    -- Ensure unique tracking per source and file
    CONSTRAINT uq_qa_file_status_source_file UNIQUE (source_key, file_path)
);

-- Create index for fast lookup during scheduling
CREATE INDEX idx_qa_file_status_lookup ON sdc_ai.qa_file_status(source_key, file_path);
CREATE INDEX idx_qa_file_status_status ON sdc_ai.qa_file_status(status);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION sdc_ai.update_qa_file_status_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = (NOW() AT TIME ZONE 'utc');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to call the update function
DROP TRIGGER IF EXISTS update_qa_file_status_timestamp ON sdc_ai.qa_file_status;
CREATE TRIGGER update_qa_file_status_timestamp
    BEFORE UPDATE ON sdc_ai.qa_file_status
    FOR EACH ROW
    EXECUTE FUNCTION sdc_ai.update_qa_file_status_timestamp();

COMMENT ON TABLE sdc_ai.qa_file_status IS 'Tracks the processing status of individual files from source systems';
COMMENT ON COLUMN sdc_ai.qa_file_status.source_key IS 'Identifier for the data source (e.g., "obs_bucket_1")';
COMMENT ON COLUMN sdc_ai.qa_file_status.file_path IS 'Full path or key of the file in the source system';
COMMENT ON COLUMN sdc_ai.qa_file_status.status IS 'Current processing status: PENDING, PROCESSING, COMPLETED, FAILED';
