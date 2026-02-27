import React, { useState } from 'react';
import axios from 'axios';

const PredictionForm: React.FC = () => {
  const [sequence, setSequence] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);

  const handleSubmit = async () => {
    const response = await axios.post('/predict', {
      sequence,
      model_type: 'quantum',
      use_quantum_hardware: false,
    });
    setJobId(response.data.job_id);
  };

  return (
    <div className="prediction-form">
      <textarea value={sequence} onChange={(e) => setSequence(e.target.value)} />
      <button onClick={handleSubmit}>Predict Structure</button>
      {jobId && <div>Job: {jobId}</div>}
    </div>
  );
};

export default PredictionForm;
