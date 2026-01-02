"""
Database initialization and migration module.

This module handles persistence schema creation, table initialization,
and migration logic for the SQLite persistence.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """
    Handles persistence schema initialization and migrations.
    
    This class is responsible for:
    - Creating all persistence tables
    - Creating indexes for performance
    - Running persistence migrations
    """
    
    def __init__(self, connection: Any, dialect: str = "sqlite"):
        """
        Initialize persistence initializer.
        
        Args:
            connection: Database persistence object (sqlite3, pymysql, etc.)
            dialect: SQL dialect ('sqlite', 'mysql', 'postgresql')
        """
        self._connection = connection
        self._dialect = dialect.lower()
        logger.debug(f"DatabaseInitializer initialized with dialect: {self._dialect}")

    def _get_pk_def(self) -> str:
        """Get primary key definition based on dialect."""
        if self._dialect == "sqlite":
            return "INTEGER PRIMARY KEY AUTOINCREMENT"
        elif self._dialect == "mysql":
            return "INTEGER PRIMARY KEY AUTO_INCREMENT"
        elif self._dialect == "postgresql":
            return "SERIAL PRIMARY KEY"
        else:
            return "INTEGER PRIMARY KEY"

    def initialize_schema(self) -> None:
        """
        Initialize persistence schema by creating all tables and indexes.
        
        This method creates all required tables for the application:
        - bots
        - conversation_history
        - custom_few_shots
        - privacy_detection_results
        - debate_sessions
        - file_uploads
        - file_parse_results
        - cached_file_analyses
        - privacy_detection_cache
        """
        cursor = self._connection.cursor()
        pk_def = self._get_pk_def()

        # Create bots table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS bots (
                bot_name TEXT PRIMARY KEY,
                persona_prompt TEXT,
                bot_configuration TEXT,
                history_mode TEXT,
                persona_mode TEXT,
                execution_mode TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'initialized',
                conversations TEXT DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create conversation_history table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id {pk_def},
                bot_name TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(bot_name, turn_index, role)
            )
        """)
        
        # Create custom_few_shots table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS custom_few_shots (
                id {pk_def},
                agent_type TEXT NOT NULL UNIQUE,
                few_shots_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create privacy_detection_results table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS privacy_detection_results (
                id {pk_def},
                detection_id TEXT NOT NULL UNIQUE,
                conversation_records TEXT NOT NULL,
                detection_result TEXT NOT NULL,
                execution_mode TEXT,
                use_few_shots BOOLEAN DEFAULT 1,
                conversation_length INTEGER,
                analyzed_at TIMESTAMP,
                agent_version TEXT,
                error TEXT,
                account_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create debate_sessions table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS debate_sessions (
                id {pk_def},
                session_id TEXT NOT NULL UNIQUE,
                topic TEXT NOT NULL,
                personas TEXT,
                agents TEXT,
                vote_history TEXT,
                reasoning_history TEXT,
                statistics TEXT,
                max_rounds INTEGER,
                current_round INTEGER DEFAULT 0,
                user_corpus TEXT,
                detected_user_style TEXT,
                debate_history TEXT,
                current_phase TEXT,
                current_round_reasonings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create file_uploads table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS file_uploads (
                id {pk_def},
                upload_id TEXT NOT NULL UNIQUE,
                account_id TEXT NOT NULL,
                message_index TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                content_type TEXT,
                file_size INTEGER,
                upload_status TEXT DEFAULT 'pending',
                parse_status TEXT DEFAULT 'pending',
                error_message TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                expires_at TIMESTAMP,
                metadata TEXT,
                UNIQUE(account_id, message_index, file_hash)
            )
        """)
        
        # Create file_parse_results table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS file_parse_results (
                id {pk_def},
                file_hash TEXT NOT NULL UNIQUE,
                parse_result TEXT NOT NULL,
                segments_count INTEGER DEFAULT 0,
                privacy_segments_count INTEGER DEFAULT 0,
                parse_mode TEXT,
                parser_used TEXT,
                confidence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Create cached_file_analyses table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS cached_file_analyses (
                id {pk_def},
                file_identifier TEXT NOT NULL,
                account_id TEXT NOT NULL,
                parse_result TEXT NOT NULL,
                message_context TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cache_version TEXT DEFAULT '1.0',
                expires_at TIMESTAMP,
                metadata TEXT,
                UNIQUE(file_identifier, account_id)
            )
        """)
        
        # Create privacy_detection_cache table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS privacy_detection_cache (
                id {pk_def},
                cache_key TEXT NOT NULL UNIQUE,
                account_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                detection_result TEXT NOT NULL,
                file_hashes TEXT,
                message_count INTEGER DEFAULT 0,
                overall_severity TEXT,
                risk_level TEXT,
                confidence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        self._create_indexes(cursor)
        
        # Run migrations
        self._run_migrations(cursor)
        
        self._connection.commit()
        logger.info("Database schema initialized successfully")
    
    def _create_indexes(self, cursor: Any) -> None:
        """Create persistence indexes for better performance."""
        indexes = [
            ("idx_conversation_bot_name", "conversation_history", "bot_name"),
            ("idx_conversation_turn", "conversation_history", "turn_index"),
            ("idx_custom_few_shots_agent_type", "custom_few_shots", "agent_type"),
            ("idx_debate_sessions_session_id", "debate_sessions", "session_id"),
            ("idx_privacy_detection_id", "privacy_detection_results", "detection_id"),
            ("idx_privacy_account_id", "privacy_detection_results", "account_id"),
            ("idx_file_uploads_account_message", "file_uploads", "account_id, message_index"),
            ("idx_file_uploads_hash", "file_uploads", "file_hash"),
            ("idx_file_uploads_status", "file_uploads", "upload_status, parse_status"),
            ("idx_file_uploads_expires", "file_uploads", "expires_at"),
            ("idx_file_parse_results_hash", "file_parse_results", "file_hash"),
            ("idx_file_parse_results_expires", "file_parse_results", "expires_at"),
            ("idx_privacy_cache_key", "privacy_detection_cache", "cache_key"),
            ("idx_privacy_cache_account", "privacy_detection_cache", "account_id"),
            ("idx_privacy_cache_session", "privacy_detection_cache", "session_id"),
            ("idx_privacy_cache_expires", "privacy_detection_cache", "expires_at"),
            ("idx_privacy_cache_access", "privacy_detection_cache", "last_accessed"),
            ("idx_privacy_analyzed_at", "privacy_detection_results", "analyzed_at"),
        ]
        
        for index_name, table, columns in indexes:
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({columns})"
            )
    
    def _run_migrations(self, cursor: Any) -> None:
        """Run persistence migrations for existing databases."""
        if self._dialect == "sqlite":
            self._migrate_debate_sessions(cursor)
            self._migrate_privacy_detection_results(cursor)
        else:
            logger.warning(f"Migrations not implemented for dialect: {self._dialect}")

    def _migrate_debate_sessions(self, cursor: Any) -> None:
        """Migrate existing debate_sessions table to support new schema."""
        try:
            cursor.execute("PRAGMA table_info(debate_sessions)")
            columns = [row[1] for row in cursor.fetchall()]
            
            new_columns = [
                ("user_corpus", "TEXT"),
                ("detected_user_style", "TEXT"),
                ("debate_history", "TEXT"),
                ("current_phase", "TEXT"),
                ("current_round_reasonings", "TEXT")
            ]
            
            for column_name, column_type in new_columns:
                if column_name not in columns:
                    logger.info(f"Adding column '{column_name}' to debate_sessions table")
                    cursor.execute(
                        f"ALTER TABLE debate_sessions ADD COLUMN {column_name} {column_type}"
                    )
            
            self._connection.commit()
            logger.info("Debate sessions migration completed")
            
        except Exception as e:
            logger.error(f"Failed to migrate debate_sessions table: {e}")
    
    def _migrate_privacy_detection_results(self, cursor: Any) -> None:
        """Migrate existing privacy_detection_results table to support account_id."""
        try:
            cursor.execute("PRAGMA table_info(privacy_detection_results)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'account_id' not in columns:
                logger.info("Adding account_id column to privacy_detection_results table")
                cursor.execute(
                    "ALTER TABLE privacy_detection_results ADD COLUMN account_id TEXT"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_privacy_account_id "
                    "ON privacy_detection_results(account_id)"
                )
            
            self._connection.commit()
            logger.info("Privacy detection results migration completed")
            
        except Exception as e:
            logger.error(f"Failed to migrate privacy_detection_results table: {e}")

