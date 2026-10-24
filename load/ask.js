import http from 'k6/http';
import { check, sleep } from 'k6';

// k6 constant-arrival-rate executor for /ask end-to-end RAG pipeline
// Tests offered arrival rates: 0.5 -> 1.0 -> 2.0 -> 5.0 -> 10.0 RPS
export const options = {
  scenarios: {
    ask_arrival_sweep: {
      executor: 'ramping-arrival-rate',
      startRate: 1,
      timeUnit: '2s', // 0.5 RPS
      preAllocatedVUs: 20,
      maxVUs: 100,
      stages: [
        { target: 1, duration: '20s' },  // 0.5 - 1 RPS
        { target: 2, duration: '20s' },  // 2 RPS
        { target: 5, duration: '20s' },  // 5 RPS
        { target: 10, duration: '20s' }, // 10 RPS
      ],
    },
  },
  thresholds: {
    // Benchmark Saturation Criteria:
    // 1. HTTP failure rate must stay below 1%
    http_req_failed: [{ threshold: 'rate<0.01', abortOnFail: false }],
    // 2. Tail latency bounds
    http_req_duration: [{ threshold: 'p(95)<2500', abortOnFail: false }],
    // 3. Generator capacity bounds
    dropped_iterations: [{ threshold: 'count<5', abortOnFail: false }],
  },
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8000';

const SAMPLE_QUESTIONS = [
  'What memory trade-offs occur during vector quantisation?',
  'What operational challenges characterize distributed intelligent software?',
  'According to statutory standards, what testing is required for government AI?',
  'How does chunk overlap influence lexical and semantic context recall?',
];

export default function () {
  const q = SAMPLE_QUESTIONS[Math.floor(Math.random() * SAMPLE_QUESTIONS.length)];
  const payload = JSON.stringify({
    question: q,
    k: 5,
    hnsw_ef: 64,
    temperature: 0.0,
    max_tokens: 128,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': `k6-ask-${__VU}-${__ITER}`,
    },
    timeout: '15s',
  };

  const res = http.post(`${BASE_URL}/ask`, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has answer': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.answer && body.answer.length > 0;
      } catch (e) {
        return false;
      }
    },
  });
}