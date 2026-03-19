db = db.getSiblingDB('eval_pipeline');

db.createCollection('conversations', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['conversation_id', 'agent_version', 'turns', 'metadata'],
      properties: {
        conversation_id: {
          bsonType: 'string',
          description: 'Unique conversation identifier'
        },
        agent_version: {
          bsonType: 'string',
          description: 'Version of the AI agent'
        },
        turns: {
          bsonType: 'array',
          items: {
            bsonType: 'object',
            required: ['turn_id', 'role', 'content', 'timestamp'],
            properties: {
              turn_id: { bsonType: 'int' },
              role: { enum: ['user', 'assistant', 'system'] },
              content: { bsonType: 'string' },
              timestamp: { bsonType: 'string' },
              tool_calls: {
                bsonType: 'array',
                items: {
                  bsonType: 'object',
                  required: ['tool_name', 'parameters'],
                  properties: {
                    tool_name: { bsonType: 'string' },
                    parameters: { bsonType: 'object' },
                    result: { bsonType: 'object' },
                    latency_ms: { bsonType: 'int' }
                  }
                }
              }
            }
          }
        },
        metadata: {
          bsonType: 'object',
          properties: {
            total_latency_ms: { bsonType: 'int' },
            mission_completed: { bsonType: 'bool' }
          }
        }
      }
    }
  }
});

db.createCollection('feedback', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['conversation_id'],
      properties: {
        conversation_id: { bsonType: 'string' },
        user_rating: { bsonType: 'int', minimum: 1, maximum: 5 },
        ops_review: {
          bsonType: 'object',
          properties: {
            quality: { enum: ['poor', 'fair', 'good', 'excellent'] },
            notes: { bsonType: 'string' }
          }
        },
        annotations: {
          bsonType: 'array',
          items: {
            bsonType: 'object',
            required: ['type', 'label', 'annotator_id'],
            properties: {
              type: { bsonType: 'string' },
              label: { bsonType: 'string' },
              annotator_id: { bsonType: 'string' },
              confidence: { bsonType: 'double' }
            }
          }
        },
        created_at: { bsonType: 'date' },
        updated_at: { bsonType: 'date' }
      }
    }
  }
});

db.createCollection('evaluations', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['evaluation_id', 'conversation_id', 'scores', 'created_at'],
      properties: {
        evaluation_id: { bsonType: 'string' },
        conversation_id: { bsonType: 'string' },
        agent_version: { bsonType: 'string' },
        eval_config_version: { bsonType: 'string' },
        scores: {
          bsonType: 'object',
          required: ['overall'],
          properties: {
            overall: { bsonType: 'double' },
            response_quality: { bsonType: 'double' },
            tool_accuracy: { bsonType: 'double' },
            coherence: { bsonType: 'double' },
            heuristic: { bsonType: 'double' }
          }
        },
        tool_evaluation: {
          bsonType: 'object',
          properties: {
            selection_accuracy: { bsonType: 'double' },
            parameter_accuracy: { bsonType: 'double' },
            hallucinated_params: { bsonType: 'array' },
            execution_success: { bsonType: 'bool' }
          }
        },
        coherence_evaluation: {
          bsonType: 'object',
          properties: {
            consistency_score: { bsonType: 'double' },
            contradiction_count: { bsonType: 'int' },
            context_retention: { bsonType: 'double' }
          }
        },
        issues_detected: {
          bsonType: 'array',
          items: {
            bsonType: 'object',
            required: ['type', 'severity', 'description'],
            properties: {
              type: { bsonType: 'string' },
              severity: { enum: ['info', 'warning', 'critical'] },
              description: { bsonType: 'string' }
            }
          }
        },
        created_at: { bsonType: 'date' }
      }
    }
  }
});

db.createCollection('suggestions', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['suggestion_id', 'type', 'suggestion', 'rationale', 'confidence', 'status'],
      properties: {
        suggestion_id: { bsonType: 'string' },
        type: { enum: ['prompt', 'tool'] },
        conversation_ids: {
          bsonType: 'array',
          items: { bsonType: 'string' }
        },
        agent_version: { bsonType: 'string' },
        suggestion: { bsonType: 'string' },
        rationale: { bsonType: 'string' },
        confidence: { bsonType: 'double' },
        expected_impact: { bsonType: 'string' },
        status: { enum: ['pending', 'applied', 'rejected', 'expired'] },
        created_at: { bsonType: 'date' },
        resolved_at: { bsonType: 'date' }
      }
    }
  }
});

db.createCollection('meta_evaluations', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['evaluator_type', 'period_start', 'period_end', 'metrics'],
      properties: {
        evaluator_type: { enum: ['llm_judge', 'tool_evaluator', 'coherence', 'heuristic'] },
        period_start: { bsonType: 'date' },
        period_end: { bsonType: 'date' },
        metrics: {
          bsonType: 'object',
          properties: {
            precision: { bsonType: 'double' },
            recall: { bsonType: 'double' },
            f1_score: { bsonType: 'double' },
            correlation_with_human: { bsonType: 'double' },
            sample_size: { bsonType: 'int' }
          }
        },
        blind_spots: {
          bsonType: 'array',
          items: {
            bsonType: 'object',
            properties: {
              category: { bsonType: 'string' },
              description: { bsonType: 'string' },
              missed_count: { bsonType: 'int' }
            }
          }
        },
        calibration_adjustments: {
          bsonType: 'object',
          properties: {
            weight_modifier: { bsonType: 'double' },
            threshold_changes: { bsonType: 'object' }
          }
        },
        created_at: { bsonType: 'date' }
      }
    }
  }
});

db.conversations.createIndex({ conversation_id: 1 }, { unique: true });
db.conversations.createIndex({ agent_version: 1 });
db.conversations.createIndex({ 'metadata.timestamp': -1 });
db.conversations.createIndex({ 'turns.tool_calls.tool_name': 1 });

db.feedback.createIndex({ conversation_id: 1 }, { unique: true });
db.feedback.createIndex({ 'annotations.annotator_id': 1 });
db.feedback.createIndex({ user_rating: 1 });
db.feedback.createIndex({ created_at: -1 });

db.evaluations.createIndex({ evaluation_id: 1 }, { unique: true });
db.evaluations.createIndex({ conversation_id: 1 });
db.evaluations.createIndex({ agent_version: 1, created_at: -1 });
db.evaluations.createIndex({ 'scores.overall': 1 });
db.evaluations.createIndex({ created_at: -1 });

db.suggestions.createIndex({ suggestion_id: 1 }, { unique: true });
db.suggestions.createIndex({ type: 1, status: 1 });
db.suggestions.createIndex({ confidence: -1 });
db.suggestions.createIndex({ agent_version: 1 });
db.suggestions.createIndex({ created_at: -1 });

db.meta_evaluations.createIndex({ evaluator_type: 1, period_end: -1 });
db.meta_evaluations.createIndex({ created_at: -1 });

print('Database initialized: collections created with validation schemas and indexes');
