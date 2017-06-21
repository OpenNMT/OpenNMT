local evaluators = {}

evaluators.Evaluator = require('onmt.evaluators.Evaluator')
evaluators.LossEvaluator = require('onmt.evaluators.LossEvaluator')
evaluators.PerplexityEvaluator = require('onmt.evaluators.PerplexityEvaluator')
evaluators.TranslationEvaluator = require('onmt.evaluators.TranslationEvaluator')
evaluators.BLEUEvaluator = require('onmt.evaluators.BLEUEvaluator')

return evaluators
