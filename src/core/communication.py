"""
Message Bus for Data Science Agent Swarm

This module provides the communication infrastructure for inter-agent messaging
using Apache Kafka or Redis as the underlying message broker.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import uuid

try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class MessageBus:
    """
    Message bus for inter-agent communication.
    
    This class provides a unified interface for sending and receiving messages
    between agents, supporting both Kafka and Redis as underlying brokers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the message bus.
        
        Args:
            config: Configuration dictionary containing broker settings
        """
        self.logger = logging.getLogger("message_bus")
        self.config = config
        self.message_handlers = {}
        self.is_running = False
        
        # Initialize broker based on configuration
        self.broker_type = config.get('broker_type', 'redis')
        self.broker = self.initialize_broker()
        
        self.logger.info(f"Message bus initialized with {self.broker_type} broker")
    
    def initialize_broker(self):
        """Initialize the message broker based on configuration."""
        if self.broker_type == 'kafka' and KAFKA_AVAILABLE:
            return self.initialize_kafka()
        elif self.broker_type == 'redis' and REDIS_AVAILABLE:
            return self.initialize_redis()
        else:
            # Fallback to in-memory broker
            self.logger.warning(f"Using in-memory broker (Kafka/Redis not available)")
            return self.initialize_memory_broker()
    
    def initialize_kafka(self):
        """Initialize Kafka broker."""
        try:
            kafka_servers = self.config.get('kafka_servers', ['localhost:9092'])
            producer = KafkaProducer(
                bootstrap_servers=kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            consumer = KafkaConsumer(
                bootstrap_servers=kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            return {
                'type': 'kafka',
                'producer': producer,
                'consumer': consumer
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka: {e}")
            return self.initialize_memory_broker()
    
    def initialize_redis(self):
        """Initialize Redis broker."""
        try:
            redis_config = self.config.get('redis', {})
            redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
            
            return {
                'type': 'redis',
                'client': redis_client
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            return self.initialize_memory_broker()
    
    def initialize_memory_broker(self):
        """Initialize in-memory broker for development/testing."""
        return {
            'type': 'memory',
            'queues': {},
            'subscribers': {}
        }
    
    async def start(self):
        """Start the message bus."""
        self.is_running = True
        self.logger.info("Message bus started")
        
        # Start message processing loop
        asyncio.create_task(self.message_processing_loop())
    
    async def stop(self):
        """Stop the message bus."""
        self.is_running = False
        
        if self.broker['type'] == 'kafka':
            self.broker['producer'].close()
            self.broker['consumer'].close()
        elif self.broker['type'] == 'redis':
            self.broker['client'].close()
        
        self.logger.info("Message bus stopped")
    
    async def publish_message(self, topic: str, message: Dict[str, Any], 
                            sender_id: str = None) -> bool:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic to publish to
            message: Message content
            sender_id: ID of the sending agent
            
        Returns:
            True if message was published successfully
        """
        try:
            # Add metadata to message
            message_with_metadata = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow().isoformat(),
                'sender_id': sender_id,
                'topic': topic,
                'data': message
            }
            
            if self.broker['type'] == 'kafka':
                future = self.broker['producer'].send(topic, message_with_metadata)
                future.get(timeout=10)  # Wait for send to complete
                
            elif self.broker['type'] == 'redis':
                self.broker['client'].publish(topic, json.dumps(message_with_metadata))
                
            elif self.broker['type'] == 'memory':
                if topic not in self.broker['queues']:
                    self.broker['queues'][topic] = []
                self.broker['queues'][topic].append(message_with_metadata)
            
            self.logger.debug(f"Message published to {topic} by {sender_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message to {topic}: {e}")
            return False
    
    async def subscribe_to_topic(self, topic: str, handler: Callable, 
                               agent_id: str = None) -> bool:
        """
        Subscribe to a topic with a message handler.
        
        Args:
            topic: Topic to subscribe to
            handler: Function to handle incoming messages
            agent_id: ID of the subscribing agent
            
        Returns:
            True if subscription was successful
        """
        try:
            subscription_id = f"{agent_id}_{topic}" if agent_id else f"anonymous_{topic}"
            self.message_handlers[subscription_id] = {
                'topic': topic,
                'handler': handler,
                'agent_id': agent_id
            }
            
            if self.broker['type'] == 'kafka':
                self.broker['consumer'].subscribe([topic])
                
            elif self.broker['type'] == 'redis':
                # Redis pub/sub is handled in the processing loop
                pass
                
            elif self.broker['type'] == 'memory':
                if topic not in self.broker['subscribers']:
                    self.broker['subscribers'][topic] = []
                self.broker['subscribers'][topic].append(subscription_id)
            
            self.logger.info(f"Agent {agent_id} subscribed to topic {topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {topic}: {e}")
            return False
    
    async def unsubscribe_from_topic(self, topic: str, agent_id: str = None) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            agent_id: ID of the unsubscribing agent
            
        Returns:
            True if unsubscription was successful
        """
        try:
            subscription_id = f"{agent_id}_{topic}" if agent_id else f"anonymous_{topic}"
            
            if subscription_id in self.message_handlers:
                del self.message_handlers[subscription_id]
            
            if self.broker['type'] == 'memory':
                if topic in self.broker['subscribers']:
                    self.broker['subscribers'][topic] = [
                        sub for sub in self.broker['subscribers'][topic] 
                        if sub != subscription_id
                    ]
            
            self.logger.info(f"Agent {agent_id} unsubscribed from topic {topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {topic}: {e}")
            return False
    
    async def message_processing_loop(self):
        """Main message processing loop."""
        while self.is_running:
            try:
                if self.broker['type'] == 'kafka':
                    await self.process_kafka_messages()
                elif self.broker['type'] == 'redis':
                    await self.process_redis_messages()
                elif self.broker['type'] == 'memory':
                    await self.process_memory_messages()
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def process_kafka_messages(self):
        """Process messages from Kafka."""
        try:
            message_batch = self.broker['consumer'].poll(timeout_ms=100)
            
            for tp, messages in message_batch.items():
                for message in messages:
                    await self.deliver_message(message.value)
                    
        except Exception as e:
            self.logger.error(f"Error processing Kafka messages: {e}")
    
    async def process_redis_messages(self):
        """Process messages from Redis pub/sub."""
        try:
            # This is a simplified implementation
            # In a real implementation, you'd use Redis pub/sub properly
            pass
        except Exception as e:
            self.logger.error(f"Error processing Redis messages: {e}")
    
    async def process_memory_messages(self):
        """Process messages from in-memory queues."""
        try:
            for topic, messages in self.broker['queues'].items():
                if topic in self.broker['subscribers']:
                    while messages:
                        message = messages.pop(0)
                        await self.deliver_message(message)
                        
        except Exception as e:
            self.logger.error(f"Error processing memory messages: {e}")
    
    async def deliver_message(self, message: Dict[str, Any]):
        """Deliver a message to all registered handlers."""
        topic = message.get('topic')
        
        for subscription_id, handler_info in self.message_handlers.items():
            if handler_info['topic'] == topic:
                try:
                    # Call the handler asynchronously
                    if asyncio.iscoroutinefunction(handler_info['handler']):
                        await handler_info['handler'](message)
                    else:
                        # Run synchronous handler in thread pool
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, handler_info['handler'], message)
                        
                except Exception as e:
                    self.logger.error(f"Error delivering message to {subscription_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get message bus status."""
        return {
            'broker_type': self.broker['type'],
            'is_running': self.is_running,
            'active_subscriptions': len(self.message_handlers),
            'topics': list(set(handler['topic'] for handler in self.message_handlers.values()))
        }
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics."""
        # This would track message counts, delivery rates, etc.
        return {
            'messages_published': 0,  # Would be tracked in real implementation
            'messages_delivered': 0,
            'delivery_rate': 1.0
        } 