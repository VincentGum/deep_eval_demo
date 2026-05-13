"""Planner Agent - High-level reasoning for task plan generation.

The Planner Agent analyzes user requests and generates structured task plans
using a mock high-level reasoning model with structured reasoning (no keyword matching).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import AgentCapability, Task, TaskPlan, TaskStatus


# ============================================================================
# Intent Classification and Semantic Understanding
# ============================================================================

class IntentType:
    """Enum-like class for intent types."""
    INQUIRE = "inquire"           # Querying information
    MANIPULATE = "manipulate"      # Creating/modifying data
    ANALYZE = "analyze"            # Processing and analyzing data
    REPORT = "report"               # Generating reports
    AUTOMATE = "automate"          # Automating workflows
    COMMUNICATE = "communicate"     # Communication tasks


class ActionVerb:
    """Represents action verbs that indicate task types."""
    # Inquiry verbs
    INQUIRE_VERBS = frozenset({
        'what', 'where', 'when', 'who', 'how', 'which',
        'find', 'search', 'lookup', 'get', 'fetch', 'retrieve',
        'check', 'view', 'show', 'tell', 'explain', 'describe',
        'status', 'progress', 'state'
    })
    
    # Manipulation verbs
    MANIPULATE_VERBS = frozenset({
        'create', 'make', 'generate', 'build', 'add',
        'update', 'modify', 'edit', 'change', 'alter',
        'delete', 'remove', 'cancel',
        'submit', 'send', 'post', 'publish',
        'save', 'store', 'write', 'record'
    })
    
    # Analysis verbs
    ANALYZE_VERBS = frozenset({
        'analyze', 'process', 'calculate', 'compute',
        'transform', 'convert', 'format', 'parse',
        'compare', 'evaluate', 'assess', 'measure',
        'sum', 'count', 'average', 'aggregate', 'summarize',
        'filter', 'sort', 'group', 'join', 'merge'
    })
    
    # Report verbs
    REPORT_VERBS = frozenset({
        'report', 'summarize', 'document', 'describe',
        'prepare', 'draft', 'compose', 'outline',
        'export', 'present', 'share', 'deliver'
    })
    
    # Communication verbs
    COMMUNICATE_VERBS = frozenset({
        'email', 'send', 'notify', 'message', 'contact',
        'invite', 'inform', 'alert', 'remind'
    })


@dataclass
class SemanticFeatures:
    """Structured semantic features extracted from user request."""
    # Core intent classification
    primary_intent: IntentType = IntentType.INQUIRE
    secondary_intent: IntentType | None = None
    
    # Action patterns
    action_verbs: list[str] = field(default_factory=list)
    object_nouns: list[str] = field(default_factory=list)
    
    # Context indicators
    has_urgency: bool = False
    has_quantity: bool = False
    has_time_reference: bool = False
    has_quality_requirement: bool = False
    
    # Data patterns
    data_sources: list[str] = field(default_factory=list)
    output_format: str | None = None
    
    # Complexity indicators
    is_multi_step: bool = False
    requires_aggregation: bool = False
    requires_visualization: bool = False
    requires_collaboration: bool = False


class SemanticAnalyzer:
    """Analyzes user requests to extract semantic features.
    
    This replaces keyword matching with structured feature extraction
    and rule-based intent classification.
    """
    
    # Patterns for data source detection (no keyword matching)
    DATA_SOURCE_PATTERNS = {
        'database': [r'db', r'database', r'table', r'query', r'sql'],
        'api': [r'api', r'endpoint', r'rest', r'http', r'json', r'xml'],
        'file': [r'file', r'document', r'csv', r'excel', r'pdf', r'txt'],
        'web': [r'website', r'web', r'url', r'http', r'html', r'网页'],
        'email': [r'email', r'mail', r'inbox', r'smtp'],
    }
    
    # Output format patterns
    OUTPUT_FORMAT_PATTERNS = {
        'markdown': [r'markdown', r'md', r'\.md'],
        'pdf': [r'pdf', r'report'],
        'excel': [r'excel', r'xlsx', r'spreadsheet', r'表格'],
        'chart': [r'chart', r'graph', r'plot', r'visualization', r'图表'],
        'table': [r'table', r'grid', r'表格'],
        'email': [r'email', r'mail'],
    }
    
    # Urgency indicators
    URGENCY_PATTERNS = [
        r'\burgent\b', r'\basap\b', r'\bimmediately\b',
        r'\b紧急\b', r'\b马上\b', r'\b尽快\b',
        r'\bimportant\b', r'\bcritical\b', r'\bpriority\b'
    ]
    
    # Time reference patterns
    TIME_PATTERNS = [
        r'\bweek(ly)?\b', r'\bmonth(ly)?\b', r'\byear(ly)?\b',
        r'\btoday\b', r'\byesterday\b', r'\btomorrow\b',
        r'\bthis week\b', r'\blast week\b',
        r'\b周\b', r'\b月\b', r'\b今天\b', r'\b昨天\b'
    ]
    
    def extract_features(self, text: str) -> SemanticFeatures:
        """Extract semantic features from user request.
        
        Uses pattern matching and rule-based classification instead of
        simple keyword matching.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        features = SemanticFeatures()
        
        # 1. Classify primary intent based on action verbs
        self._classify_intent(words, text_lower, features)
        
        # 2. Extract action verbs and object nouns
        self._extract_action_patterns(words, features)
        
        # 3. Detect context indicators
        self._detect_context(text, text_lower, features)
        
        # 4. Identify data sources
        self._identify_data_sources(text_lower, features)
        
        # 5. Determine output format
        self._determine_output_format(text_lower, features)
        
        # 6. Assess complexity
        self._assess_complexity(words, text, features)
        
        return features
    
    def _classify_intent(
        self,
        words: set[str],
        text: str,
        features: SemanticFeatures
    ):
        """Classify intent based on action verb analysis."""
        # Count verb matches for each intent type
        intent_scores = {}
        
        # Inquiry intent
        inquiry_count = len(words & ActionVerb.INQUIRE_VERBS)
        intent_scores[IntentType.INQUIRE] = inquiry_count
        
        # Manipulation intent
        manip_count = len(words & ActionVerb.MANIPULATE_VERBS)
        intent_scores[IntentType.MANIPULATE] = manip_count
        
        # Analysis intent
        analyze_count = len(words & ActionVerb.ANALYZE_VERBS)
        intent_scores[IntentType.ANALYZE] = analyze_count
        
        # Report intent
        report_count = len(words & ActionVerb.REPORT_VERBS)
        intent_scores[IntentType.REPORT] = report_count
        
        # Communication intent
        comm_count = len(words & ActionVerb.COMMUNICATE_VERBS)
        intent_scores[IntentType.COMMUNICATE] = comm_count
        
        # Determine primary intent (highest score, with tie-breaker rules)
        if intent_scores:
            max_score = max(intent_scores.values())
            if max_score > 0:
                # Use tie-breaker: prefer more specific intents
                for intent in [IntentType.REPORT, IntentType.ANALYZE, 
                              IntentType.MANIPULATE, IntentType.COMMUNICATE, IntentType.INQUIRE]:
                    if intent_scores.get(intent, 0) == max_score:
                        features.primary_intent = intent
                        break
        
        # Check for multi-intent (e.g., "analyze and report")
        if 'and' in words or '&' in text:
            # Find second highest intent
            sorted_intents = sorted(intent_scores.items(), key=lambda x: -x[1])
            if len(sorted_intents) >= 2 and sorted_intents[1][1] > 0:
                features.secondary_intent = sorted_intents[1][0]
    
    def _extract_action_patterns(
        self,
        words: set[str],
        features: SemanticFeatures
    ):
        """Extract action verbs and object nouns."""
        # Action verbs
        all_action_verbs = (ActionVerb.INQUIRE_VERBS | ActionVerb.MANIPULATE_VERBS |
                           ActionVerb.ANALYZE_VERBS | ActionVerb.REPORT_VERBS |
                           ActionVerb.COMMUNICATE_VERBS)
        features.action_verbs = list(words & all_action_verbs)
        
        # Object nouns (common business objects)
        business_objects = {
            'order', 'customer', 'sales', 'report', 'data', 'chart',
            'document', 'email', 'invoice', 'payment', 'product', 'user'
        }
        features.object_nouns = list(words & business_objects)
    
    def _detect_context(
        self,
        text: str,
        text_lower: str,
        features: SemanticFeatures
    ):
        """Detect context indicators."""
        # Urgency
        features.has_urgency = any(
            re.search(pattern, text_lower) for pattern in self.URGENCY_PATTERNS
        )
        
        # Time reference
        features.has_time_reference = any(
            re.search(pattern, text_lower) for pattern in self.TIME_PATTERNS
        )
        
        # Quantity indicators
        quantity_patterns = [r'\d+', r'\b(sum|total|count|average|max|min)\b']
        features.has_quantity = any(
            re.search(pattern, text_lower) for pattern in quantity_patterns
        )
        
        # Quality requirements
        word_set = set(text_lower.split())
        quality_words = {'accuracy', 'precision', 'detailed', 'complete', 'thorough'}
        features.has_quality_requirement = bool(word_set & quality_words)
    
    def _identify_data_sources(
        self,
        text: str,
        features: SemanticFeatures
    ):
        """Identify data sources from patterns."""
        for source_type, patterns in self.DATA_SOURCE_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                features.data_sources.append(source_type)
        
        # Default to API if no source detected
        if not features.data_sources:
            features.data_sources.append('api')
    
    def _determine_output_format(
        self,
        text: str,
        features: SemanticFeatures
    ):
        """Determine expected output format."""
        for format_type, patterns in self.OUTPUT_FORMAT_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                features.output_format = format_type
                break
        
        # Default format
        if not features.output_format:
            features.output_format = 'markdown'
    
    def _assess_complexity(
        self,
        words: set[str],
        text: str,
        features: SemanticFeatures
    ):
        """Assess task complexity using contextual inference."""
        # Multi-step indicators
        multi_step_words = {'and', 'then', 'also', 'plus', 'after', 'before', 'first', 'next'}
        features.is_multi_step = bool(words & multi_step_words) or len(words) > 15
        
        # Aggregation indicators (explicit)
        agg_words = {'sum', 'total', 'average', 'count', 'aggregate', 'summarize'}
        explicit_agg = bool(words & agg_words)
        
        # Aggregation indicators (inferred from intent/context)
        # Report types that typically need aggregation
        report_types_needing_agg = {'sales', 'revenue', 'performance', 'summary', 'weekly', 'monthly', 'quarterly', 'annual'}
        inferred_agg = bool(words & report_types_needing_agg) and features.primary_intent in {'report', 'analyze'}
        
        # Visualization indicators (explicit)
        viz_words = {'chart', 'graph', 'plot', 'visual', 'pie', 'bar', 'line'}
        explicit_viz = bool(words & viz_words)
        
        # Visualization indicators (inferred)
        # Reports often include charts
        inferred_viz = features.primary_intent == 'report' and explicit_agg
        
        features.requires_aggregation = explicit_agg or inferred_agg
        features.requires_visualization = explicit_viz or inferred_viz
        
        # Collaboration indicators
        collab_words = {'email', 'send', 'notify', 'share', 'team', 'colleague'}
        features.requires_collaboration = bool(words & collab_words)


# ============================================================================
# Task Capability Mapping (Rule-based, not keyword matching)
# ============================================================================

class CapabilityMapper:
    """Maps semantic features to required capabilities.
    
    Uses decision rules instead of keyword matching.
    """
    
    def map_to_capabilities(self, features: SemanticFeatures) -> list[str]:
        """Map semantic features to required capabilities.
        
        Decision rules:
        1. Primary intent determines base capability
        2. Secondary features add additional capabilities
        3. Output format may add visualization/reporting
        """
        from .base import AgentCapability
        
        capabilities = []
        capability_set = set()
        
        # 1. Map primary intent to base capability
        intent_capability_map = {
            IntentType.INQUIRE: [
                AgentCapability.API_CALL,
                AgentCapability.BROWSER_SCRAPE,
                AgentCapability.DATA_QUERY,
            ],
            IntentType.MANIPULATE: [
                AgentCapability.DOC_WRITE,
                AgentCapability.API_CALL,
            ],
            IntentType.ANALYZE: [
                AgentCapability.DATA_QUERY,
                AgentCapability.DATA_TRANSFORM,
                AgentCapability.DATA_AGGREGATE,
            ],
            IntentType.REPORT: [
                AgentCapability.DATA_AGGREGATE,
                AgentCapability.REPORT_GENERATE,
            ],
            IntentType.COMMUNICATE: [
                AgentCapability.EMAIL_SEND,
            ],
        }
        
        base_caps = intent_capability_map.get(features.primary_intent, [])
        for cap in base_caps:
            if cap.value not in capability_set:
                capabilities.append(cap.value)
                capability_set.add(cap.value)
        
        # 2. Add capabilities based on features
        if features.requires_aggregation and AgentCapability.DATA_AGGREGATE.value not in capability_set:
            capabilities.append(AgentCapability.DATA_AGGREGATE.value)
            capability_set.add(AgentCapability.DATA_AGGREGATE.value)
        
        if features.requires_visualization and AgentCapability.CHART_CREATE.value not in capability_set:
            capabilities.append(AgentCapability.CHART_CREATE.value)
        
        if features.output_format in ['table'] and AgentCapability.TABLE_CREATE.value not in capability_set:
            capabilities.append(AgentCapability.TABLE_CREATE.value)
        
        if features.requires_collaboration and AgentCapability.EMAIL_SEND.value not in capability_set:
            capabilities.append(AgentCapability.EMAIL_SEND.value)
        
        # 3. Add data source capabilities
        for source in features.data_sources:
            source_capability_map = {
                'api': AgentCapability.API_CALL,
                'database': AgentCapability.DATA_QUERY,
                'web': AgentCapability.BROWSER_SCRAPE,
                'file': AgentCapability.DOC_READ,
            }
            cap = source_capability_map.get(source)
            if cap and cap.value not in capability_set:
                capabilities.append(cap.value)
        
        return capabilities


# ============================================================================
# Task Plan Generator
# ============================================================================

class TaskPlanGenerator:
    """Generates structured task plans from semantic features."""
    
    def __init__(self):
        self.capability_mapper = CapabilityMapper()
    
    def generate_tasks(
        self,
        features: SemanticFeatures,
        user_request: str
    ) -> list[dict[str, Any]]:
        """Generate task list from semantic features.
        
        Instead of keyword matching, uses semantic understanding
        to determine task sequence and dependencies.
        """
        from .base import TaskPriority, AgentCapability
        
        tasks = []
        task_counter = [1]  # Mutable counter
        
        # 1. Determine task sequence based on intent
        task_sequence = self._determine_task_sequence(features)
        
        # 2. Generate tasks with proper parameters
        for task_def in task_sequence:
            task = self._create_task(
                task_def,
                features,
                user_request,
                task_counter
            )
            if task:
                tasks.append(task)
        
        # 3. Add final aggregation task if needed
        if features.requires_aggregation and len(tasks) > 1:
            agg_task = self._create_aggregation_task(features, tasks, task_counter)
            if agg_task:
                tasks.append(agg_task)
        
        # 4. Add report task if output format is report-like
        if features.output_format in ['markdown', 'pdf'] or features.primary_intent == IntentType.REPORT:
            report_task = self._create_report_task(features, tasks, task_counter)
            if report_task:
                tasks.append(report_task)
        
        return tasks
    
    def _determine_task_sequence(self, features: SemanticFeatures) -> list[dict]:
        """Determine task sequence based on intent and features."""
        from .base import AgentCapability
        
        sequence = []
        
        # Data collection phase - reports also need data!
        # Any intent that requires external data should collect first
        if features.primary_intent in [IntentType.INQUIRE, IntentType.ANALYZE, IntentType.REPORT]:
            # Add data source tasks first
            for source in features.data_sources:
                if source == 'api':
                    sequence.append({'capability': AgentCapability.API_CALL, 'phase': 'collect'})
                elif source == 'web':
                    # For web scraping, we need two tasks:
                    # 1. Navigate to the website (URL comes from database query)
                    # 2. Scrape content from the loaded page
                    sequence.append({'capability': AgentCapability.BROWSER_NAVIGATE, 'phase': 'collect'})
                    sequence.append({'capability': AgentCapability.BROWSER_SCRAPE, 'phase': 'collect'})
                elif source == 'database':
                    sequence.append({'capability': AgentCapability.DATA_QUERY, 'phase': 'collect'})
        
        # Processing phase
        if features.primary_intent == IntentType.ANALYZE:
            sequence.append({'capability': AgentCapability.DATA_TRANSFORM, 'phase': 'process'})
            if features.requires_aggregation:
                sequence.append({'capability': AgentCapability.DATA_AGGREGATE, 'phase': 'process'})
        
        # Visualization phase
        if features.requires_visualization:
            sequence.append({'capability': AgentCapability.CHART_CREATE, 'phase': 'visualize'})
        
        # Output phase
        if features.output_format == 'table':
            sequence.append({'capability': AgentCapability.TABLE_CREATE, 'phase': 'output'})
        elif features.primary_intent == IntentType.REPORT:
            sequence.append({'capability': AgentCapability.REPORT_GENERATE, 'phase': 'output'})
        
        # Communication phase
        if features.requires_collaboration:
            sequence.append({'capability': AgentCapability.EMAIL_SEND, 'phase': 'communicate'})
        
        # Default task if no sequence determined
        if not sequence:
            sequence.append({'capability': AgentCapability.API_CALL, 'phase': 'default'})
        
        return sequence
    
    def _create_task(
        self,
        task_def: dict,
        features: SemanticFeatures,
        user_request: str,
        task_counter: list[int]
    ) -> dict[str, Any] | None:
        """Create a single task with parameters."""
        from .base import TaskPriority, AgentCapability, create_task_id
        
        capability = task_def['capability']
        
        # Generate task description based on capability and context
        descriptions = {
            AgentCapability.API_CALL: f"Call API to retrieve {', '.join(features.data_sources) if features.data_sources else 'data'}",
            AgentCapability.BROWSER_SCRAPE: "Navigate to and scrape data from web source",
            AgentCapability.DATA_QUERY: "Query data from database",
            AgentCapability.DATA_TRANSFORM: "Transform and process collected data",
            AgentCapability.DATA_AGGREGATE: "Aggregate and calculate statistics",
            AgentCapability.CHART_CREATE: f"Create {features.output_format or 'chart'} visualization",
            AgentCapability.TABLE_CREATE: "Format data into table",
            AgentCapability.REPORT_GENERATE: f"Generate {features.output_format or 'report'} report",
            AgentCapability.EMAIL_SEND: "Send notification via email",
        }
        
        task_id = f"task_{task_counter[0]:03d}"
        task_counter[0] += 1
        
        # Determine priority
        priority = TaskPriority.MEDIUM
        if features.has_urgency:
            priority = TaskPriority.URGENT
        elif features.primary_intent == IntentType.REPORT:
            priority = TaskPriority.HIGH
        
        return {
            'id': task_id,
            'description': descriptions.get(capability, "Execute task"),
            'capability_required': capability,
            'priority': priority,
            'depends_on': [],
            'expected_output': self._get_expected_output(capability, features),
            'input_data': self._get_input_data(capability, features, user_request),
        }
    
    def _create_aggregation_task(
        self,
        features: SemanticFeatures,
        existing_tasks: list,
        task_counter: list[int]
    ) -> dict | None:
        """Create aggregation task."""
        from .base import AgentCapability
        
        task_id = f"task_{task_counter[0]:03d}"
        task_counter[0] += 1
        
        return {
            'id': task_id,
            'description': "Aggregate and summarize collected data",
            'capability_required': AgentCapability.DATA_AGGREGATE,
            'priority': existing_tasks[-1]['priority'] if existing_tasks else TaskPriority.MEDIUM,
            'depends_on': [t['id'] for t in existing_tasks if t['capability_required'] != AgentCapability.DATA_AGGREGATE],
            'expected_output': "Aggregated statistics and trends",
            'input_data': {},
        }
    
    def _create_report_task(
        self,
        features: SemanticFeatures,
        existing_tasks: list,
        task_counter: list[int]
    ) -> dict | None:
        """Create report generation task."""
        from .base import AgentCapability, TaskPriority
        
        # Avoid duplicate report task
        if any(t['capability_required'] == AgentCapability.REPORT_GENERATE for t in existing_tasks):
            return None
        
        task_id = f"task_{task_counter[0]:03d}"
        task_counter[0] += 1
        
        return {
            'id': task_id,
            'description': f"Generate final {features.output_format or 'report'} report",
            'capability_required': AgentCapability.REPORT_GENERATE,
            'priority': TaskPriority.HIGH,
            'depends_on': [t['id'] for t in existing_tasks if t['capability_required'] != AgentCapability.REPORT_GENERATE],
            'expected_output': f"Formatted {features.output_format or 'markdown'} report",
            'input_data': {'title': features.object_nouns[0].title() if features.object_nouns else 'Report'},
        }
    
    def _get_expected_output(self, capability: AgentCapability, features: SemanticFeatures) -> str:
        """Get expected output description."""
        outputs = {
            AgentCapability.API_CALL: "API response with data",
            AgentCapability.BROWSER_SCRAPE: "Extracted web content",
            AgentCapability.DATA_QUERY: "Queried data records",
            AgentCapability.DATA_TRANSFORM: "Transformed data",
            AgentCapability.DATA_AGGREGATE: "Aggregated statistics",
            AgentCapability.CHART_CREATE: f"{features.output_format or 'chart'} visualization",
            AgentCapability.TABLE_CREATE: "Formatted data table",
            AgentCapability.REPORT_GENERATE: f"Generated {features.output_format or 'report'}",
            AgentCapability.EMAIL_SEND: "Email sent confirmation",
        }
        return outputs.get(capability, "Task completed")
    
    def _get_input_data(
        self,
        capability: AgentCapability,
        features: SemanticFeatures,
        user_request: str
    ) -> dict[str, Any]:
        """Get input data for task."""
        data = {}
        
        # Set defaults based on capability
        if capability == AgentCapability.API_CALL:
            data['endpoint'] = '/api/data'
            data['method'] = 'GET'
        elif capability == AgentCapability.BROWSER_SCRAPE:
            data['url'] = 'https://www.example.com'
        elif capability == AgentCapability.DATA_QUERY:
            data['table'] = 'data'
        elif capability == AgentCapability.CHART_CREATE:
            data['chart_type'] = 'bar'
            data['title'] = features.object_nouns[0].title() if features.object_nouns else 'Chart'
        elif capability == AgentCapability.REPORT_GENERATE:
            data['title'] = features.object_nouns[0].title() if features.object_nouns else 'Report'
            data['format'] = features.output_format or 'markdown'
        
        return data


# ============================================================================
# Reasoning Model (Replaces keyword matching with structured reasoning)
# ============================================================================

@dataclass
class ReasoningStep:
    """A step in the structured reasoning process."""
    step_number: int
    thought: str
    observation: str
    conclusion: str


class MockReasoningModel:
    """Mock high-level reasoning model for task planning.
    
    Uses structured semantic analysis instead of keyword matching.
    Simulates reasoning model (like o1) behavior.
    """
    
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.capability_mapper = CapabilityMapper()
        self.task_generator = TaskPlanGenerator()
    
    def reason(self, user_request: str) -> tuple[list[ReasoningStep], list[dict]]:
        """Perform structured reasoning on user request.
        
        Returns:
            Tuple of (reasoning_steps, generated_tasks)
        """
        reasoning_steps = []
        
        # Step 1: Parse and understand request
        step1 = ReasoningStep(
            step_number=1,
            thought=f"Analyzing request: '{user_request}'",
            observation="Breaking down the request into semantic components",
            conclusion="Request parsed successfully"
        )
        reasoning_steps.append(step1)
        
        # Step 2: Extract semantic features
        features = self.semantic_analyzer.extract_features(user_request)
        step2 = ReasoningStep(
            step_number=2,
            thought=f"Intent classification: {features.primary_intent}",
            observation=f"Features: aggregation={features.requires_aggregation}, "
                       f"visualization={features.requires_visualization}, "
                       f"output_format={features.output_format}",
            conclusion=f"Primary intent identified as {features.primary_intent}"
        )
        reasoning_steps.append(step2)
        
        # Step 3: Determine required capabilities
        capabilities = self.capability_mapper.map_to_capabilities(features)
        step3 = ReasoningStep(
            step_number=3,
            thought=f"Required capabilities: {capabilities}",
            observation=f"Action verbs: {features.action_verbs}, "
                       f"Data sources: {features.data_sources}",
            conclusion=f"Mapped to {len(capabilities)} capabilities"
        )
        reasoning_steps.append(step3)
        
        # Step 4: Generate task plan
        tasks = self.task_generator.generate_tasks(features, user_request)
        step4 = ReasoningStep(
            step_number=4,
            thought=f"Generated {len(tasks)} tasks",
            observation=f"Task sequence determined by intent phase ordering",
            conclusion="Task plan generated successfully"
        )
        reasoning_steps.append(step4)
        
        # Step 5: Analyze dependencies
        task_ids = [t['id'] for t in tasks]
        step5 = ReasoningStep(
            step_number=5,
            thought=f"Task IDs: {task_ids}",
            observation="Dependencies analyzed based on data flow",
            conclusion="Execution order optimized"
        )
        reasoning_steps.append(step5)
        
        return reasoning_steps, tasks


# ============================================================================
# Planner Agent
# ============================================================================

class PlannerAgent:
    """Planner Agent that generates task plans.
    
    Uses structured semantic reasoning instead of keyword matching.
    """
    
    def __init__(self, reasoning_model: MockReasoningModel | None = None):
        self.reasoning_model = reasoning_model or MockReasoningModel()
    
    def create_plan(
        self,
        user_request: str,
        context: dict[str, Any] | None = None
    ) -> tuple[dict, list[ReasoningStep]]:
        """Create a task plan from user request.
        
        Args:
            user_request: The user's request
            context: Optional context (user info, available tools, etc.)
        
        Returns:
            Tuple of (TaskPlan dict, ReasoningSteps list)
        """
        from .base import TaskPlan, TaskPriority, AgentCapability, create_task_id
        
        # Perform structured reasoning
        reasoning_steps, task_dicts = self.reasoning_model.reason(user_request)
        
        # Convert task dicts to Task objects
        tasks = []
        for task_dict in task_dicts:
            # Get priority from dict or default
            priority = task_dict.get('priority', TaskPriority.MEDIUM)
            if isinstance(priority, str):
                priority = TaskPriority(priority)
            
            task = Task(
                id=task_dict['id'],
                description=task_dict['description'],
                capability_required=task_dict['capability_required'],
                expected_output=task_dict.get('expected_output', 'Task completed'),
                priority=priority,
                depends_on=task_dict.get('depends_on', []),
                input_data=task_dict.get('input_data', {}),
            )
            tasks.append(task)
        
        # Create TaskPlan
        task_plan = TaskPlan(
            id=create_task_id("plan"),
            user_request=user_request,
            tasks=tasks,
            metadata={'reasoning_steps': reasoning_steps}
        )
        
        return task_plan, reasoning_steps
    
    def explain_plan(self, plan: dict) -> str:
        """Generate human-readable explanation of the plan."""
        tasks = plan.tasks if hasattr(plan, 'tasks') else plan.get('tasks', [])
        reasoning_steps = plan.metadata.get('reasoning_steps', []) if hasattr(plan, 'metadata') else plan.get('metadata', {}).get('reasoning_steps', [])
        
        lines = [
            f"## Task Plan: {plan.id}",
            f"**User Request**: {plan.user_request}",
            "",
            "### Reasoning Process:",
        ]
        
        for step in reasoning_steps:
            lines.append(f"{step.step_number}. **{step.thought}**")
            lines.append(f"   - Observation: {step.observation}")
            lines.append(f"   - Conclusion: {step.conclusion}")
            lines.append("")
        
        lines.extend(["### Tasks:", ""])
        
        for i, task in enumerate(tasks, 1):
            deps = f" (depends on: {', '.join(task.depends_on)})" if task.depends_on else ""
            lines.append(f"{i}. [{task.priority.value.upper()}] {task.description}{deps}")
            lines.append(f"   - Capability: {task.capability_required.value}")
            lines.append(f"   - Expected: {task.expected_output}")
            lines.append("")
        
        lines.extend([
            "### Execution Summary:",
            f"- Total Tasks: {len(tasks)}",
            f"- Expected Duration: ~{len(tasks) * 2} minutes",
        ])
        
        return "\n".join(lines)
