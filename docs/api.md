# QuantumFold-Advantage API

## Endpoints
- `POST /predict`
- `GET /status/{job_id}`
- `GET /result/{job_id}`
- `POST /batch_predict`
- `GET /models`

## Example
```bash
curl -X POST http://localhost:8000/predict -H 'content-type: application/json' -d '{"sequence":"ACDEFGHIK"}'
```

## Rate limits
- Free tier: 10 predictions/day
- Pro tier: unlimited
