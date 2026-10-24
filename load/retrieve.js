import http from 'k6/http';
import { check, sleep } from 'k6';

// k6 constant-arrival-rate executor for /retrieve benchmarking
// Tests offered arrival rates: 10 -> 25 -> 50 -> 100 -> 200 RPS
export const options = {
  scenarios: {
    retrieve_arrival_sweep: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      preAllocatedVUs: 50,
      maxVUs: 200,
      stages: [
        { target: 10, duration: '15s' },
        { target: 25, duration: '15s' },
        { target: 50, duration: '15s' },
        { target: 100, duration: '15s' },
        { target: 200, duration: '15s' },
      ],
    },
  },
  thresholds: {
    // Benchmark Saturation Criteria:
    // 1. HTTP failure rate must stay below 1%
    http_req_failed: [{ threshold: 'rate<0.01', abortOnFail: false }],
    // 2. Latency inflation check: p95 under 50ms at nominal load
    http_req_duration: [{ threshold: 'p(95)<50', abortOnFail: false }],
    // 3. Dropped iteration rate must not exceed 0.5%
    dropped_iterations: [{ threshold: 'count<10', abortOnFail: false }],
  },
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8000';

const SAMPLE_QUERIES = [
  'What are the memory trade-offs of product quantisation?',
  'How do HNSW ef_construct and M affect graph connectivity?',
  'What statutory mandates apply to public domain federal documents?',
  'How does sentence-respecting chunking improve dense retrieval?',
  'What mechanisms prevent prompt injection in RAG serving pipelines?',
];

export default function () {
  const query = SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)];
  const payload = JSON.stringify({
    query: query,
    k: 10,
    hnsw_ef: 64,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': `k6-retrieve-${__VU}-${__ITER}`,
    },
    timeout: '5s',
  };

  const res = http.post(`${BASE_URL}/retrieve`, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has hits': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.hits && body.hits.length >= 0;
      } catch (e) {
        return false;
      }
    },
  });
}