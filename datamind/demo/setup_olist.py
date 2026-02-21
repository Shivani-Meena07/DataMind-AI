"""
Olist E-Commerce Database Setup

This script creates a SQLite database with the Olist E-Commerce schema
for demonstration purposes. It generates sample data that mirrors
the actual Olist dataset structure.
"""

import sqlite3
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
import hashlib


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID string."""
    chars = string.ascii_lowercase + string.digits
    random_str = ''.join(random.choices(chars, k=32))
    return hashlib.md5(f"{prefix}{random_str}".encode()).hexdigest()


def random_date(start_year: int = 2017, end_year: int = 2018) -> datetime:
    """Generate a random date within range."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def setup_olist_database(db_path: str = "./data/olist.db"):
    """
    Create and populate the Olist E-Commerce database.
    
    The Olist dataset contains 9 main tables:
    1. olist_customers - Customer information
    2. olist_geolocation - Brazilian zip codes with lat/lng
    3. olist_order_items - Items within each order
    4. olist_order_payments - Payment information
    5. olist_order_reviews - Customer reviews
    6. olist_orders - Order information
    7. olist_products - Product catalog
    8. olist_sellers - Seller information
    9. product_category_name_translation - Category translations
    """
    
    # Create directory if needed
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating Olist E-Commerce Database Schema...")
    
    # Drop existing tables
    tables = [
        'olist_order_reviews',
        'olist_order_payments', 
        'olist_order_items',
        'olist_orders',
        'olist_products',
        'olist_sellers',
        'olist_customers',
        'olist_geolocation',
        'product_category_name_translation',
    ]
    
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    
    # Create tables
    
    # 1. Geolocation table
    cursor.execute("""
        CREATE TABLE olist_geolocation (
            geolocation_zip_code_prefix TEXT,
            geolocation_lat REAL,
            geolocation_lng REAL,
            geolocation_city TEXT,
            geolocation_state TEXT
        )
    """)
    
    # 2. Customers table
    cursor.execute("""
        CREATE TABLE olist_customers (
            customer_id TEXT PRIMARY KEY,
            customer_unique_id TEXT,
            customer_zip_code_prefix TEXT,
            customer_city TEXT,
            customer_state TEXT
        )
    """)
    
    # 3. Sellers table
    cursor.execute("""
        CREATE TABLE olist_sellers (
            seller_id TEXT PRIMARY KEY,
            seller_zip_code_prefix TEXT,
            seller_city TEXT,
            seller_state TEXT
        )
    """)
    
    # 4. Product Category Translation table
    cursor.execute("""
        CREATE TABLE product_category_name_translation (
            product_category_name TEXT PRIMARY KEY,
            product_category_name_english TEXT
        )
    """)
    
    # 5. Products table
    cursor.execute("""
        CREATE TABLE olist_products (
            product_id TEXT PRIMARY KEY,
            product_category_name TEXT,
            product_name_length INTEGER,
            product_description_length INTEGER,
            product_photos_qty INTEGER,
            product_weight_g REAL,
            product_length_cm REAL,
            product_height_cm REAL,
            product_width_cm REAL,
            FOREIGN KEY (product_category_name) 
                REFERENCES product_category_name_translation(product_category_name)
        )
    """)
    
    # 6. Orders table
    cursor.execute("""
        CREATE TABLE olist_orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            order_status TEXT,
            order_purchase_timestamp TEXT,
            order_approved_at TEXT,
            order_delivered_carrier_date TEXT,
            order_delivered_customer_date TEXT,
            order_estimated_delivery_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES olist_customers(customer_id)
        )
    """)
    
    # 7. Order Items table
    cursor.execute("""
        CREATE TABLE olist_order_items (
            order_id TEXT,
            order_item_id INTEGER,
            product_id TEXT,
            seller_id TEXT,
            shipping_limit_date TEXT,
            price REAL,
            freight_value REAL,
            PRIMARY KEY (order_id, order_item_id),
            FOREIGN KEY (order_id) REFERENCES olist_orders(order_id),
            FOREIGN KEY (product_id) REFERENCES olist_products(product_id),
            FOREIGN KEY (seller_id) REFERENCES olist_sellers(seller_id)
        )
    """)
    
    # 8. Order Payments table
    cursor.execute("""
        CREATE TABLE olist_order_payments (
            order_id TEXT,
            payment_sequential INTEGER,
            payment_type TEXT,
            payment_installments INTEGER,
            payment_value REAL,
            PRIMARY KEY (order_id, payment_sequential),
            FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
        )
    """)
    
    # 9. Order Reviews table
    cursor.execute("""
        CREATE TABLE olist_order_reviews (
            review_id TEXT PRIMARY KEY,
            order_id TEXT,
            review_score INTEGER,
            review_comment_title TEXT,
            review_comment_message TEXT,
            review_creation_date TEXT,
            review_answer_timestamp TEXT,
            FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
        )
    """)
    
    print("Schema created. Populating with sample data...")
    
    # Populate with sample data
    
    # Brazilian states and cities
    locations = [
        ('01310', -23.5505, -46.6333, 'sao paulo', 'SP'),
        ('20040', -22.9068, -43.1729, 'rio de janeiro', 'RJ'),
        ('30130', -19.9167, -43.9345, 'belo horizonte', 'MG'),
        ('80010', -25.4284, -49.2733, 'curitiba', 'PR'),
        ('90010', -30.0346, -51.2177, 'porto alegre', 'RS'),
        ('40010', -12.9714, -38.5014, 'salvador', 'BA'),
        ('60010', -3.7172, -38.5433, 'fortaleza', 'CE'),
        ('70040', -15.7942, -47.8822, 'brasilia', 'DF'),
        ('66010', -1.4558, -48.4902, 'belem', 'PA'),
        ('69010', -3.1190, -60.0217, 'manaus', 'AM'),
    ]
    
    # Insert geolocation data
    for loc in locations:
        cursor.execute("""
            INSERT INTO olist_geolocation VALUES (?, ?, ?, ?, ?)
        """, loc)
    
    # Product categories
    categories = [
        ('informatica_acessorios', 'computers_accessories'),
        ('telefonia', 'telephony'),
        ('utilidades_domesticas', 'housewares'),
        ('esporte_lazer', 'sports_leisure'),
        ('moveis_decoracao', 'furniture_decor'),
        ('beleza_saude', 'health_beauty'),
        ('cama_mesa_banho', 'bed_bath_table'),
        ('eletronicos', 'electronics'),
        ('fashion_bolsas_e_acessorios', 'fashion_bags_accessories'),
        ('brinquedos', 'toys'),
        ('automotivo', 'auto'),
        ('livros_interesse_geral', 'books_general_interest'),
        ('cool_stuff', 'cool_stuff'),
        ('ferramentas_jardim', 'garden_tools'),
        ('perfumaria', 'perfumery'),
    ]
    
    for cat in categories:
        cursor.execute("""
            INSERT INTO product_category_name_translation VALUES (?, ?)
        """, cat)
    
    # Generate customers (500)
    print("  Generating customers...")
    customer_ids = []
    for i in range(500):
        customer_id = generate_id(f"cust_{i}")
        customer_unique_id = generate_id(f"uniq_{i}")
        loc = random.choice(locations)
        
        customer_ids.append(customer_id)
        cursor.execute("""
            INSERT INTO olist_customers VALUES (?, ?, ?, ?, ?)
        """, (customer_id, customer_unique_id, loc[0], loc[3], loc[4]))
    
    # Generate sellers (100)
    print("  Generating sellers...")
    seller_ids = []
    for i in range(100):
        seller_id = generate_id(f"seller_{i}")
        loc = random.choice(locations)
        
        seller_ids.append(seller_id)
        cursor.execute("""
            INSERT INTO olist_sellers VALUES (?, ?, ?, ?)
        """, (seller_id, loc[0], loc[3], loc[4]))
    
    # Generate products (300)
    print("  Generating products...")
    product_ids = []
    for i in range(300):
        product_id = generate_id(f"prod_{i}")
        category = random.choice(categories)[0]
        
        product_ids.append(product_id)
        cursor.execute("""
            INSERT INTO olist_products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product_id,
            category,
            random.randint(20, 100),  # name length
            random.randint(100, 2000),  # description length
            random.randint(1, 10),  # photos
            random.uniform(100, 5000),  # weight
            random.uniform(10, 100),  # length
            random.uniform(5, 50),  # height
            random.uniform(5, 50),  # width
        ))
    
    # Generate orders (1000)
    print("  Generating orders...")
    order_ids = []
    order_statuses = ['delivered', 'delivered', 'delivered', 'delivered', 
                      'shipped', 'processing', 'canceled', 'unavailable']
    
    for i in range(1000):
        order_id = generate_id(f"order_{i}")
        customer_id = random.choice(customer_ids)
        status = random.choice(order_statuses)
        
        purchase_date = random_date()
        approved_date = purchase_date + timedelta(hours=random.randint(1, 24))
        carrier_date = approved_date + timedelta(days=random.randint(1, 5)) if status in ['delivered', 'shipped'] else None
        delivered_date = carrier_date + timedelta(days=random.randint(1, 10)) if status == 'delivered' else None
        estimated_date = purchase_date + timedelta(days=random.randint(7, 30))
        
        order_ids.append(order_id)
        cursor.execute("""
            INSERT INTO olist_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_id,
            customer_id,
            status,
            purchase_date.isoformat(),
            approved_date.isoformat() if approved_date else None,
            carrier_date.isoformat() if carrier_date else None,
            delivered_date.isoformat() if delivered_date else None,
            estimated_date.isoformat(),
        ))
    
    # Generate order items (2000)
    print("  Generating order items...")
    for order_id in order_ids:
        num_items = random.randint(1, 5)
        for item_id in range(1, num_items + 1):
            product_id = random.choice(product_ids)
            seller_id = random.choice(seller_ids)
            price = round(random.uniform(20, 500), 2)
            freight = round(random.uniform(10, 100), 2)
            
            cursor.execute("""
                INSERT INTO olist_order_items VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id,
                item_id,
                product_id,
                seller_id,
                (random_date() + timedelta(days=3)).isoformat(),
                price,
                freight,
            ))
    
    # Generate order payments (1000)
    print("  Generating order payments...")
    payment_types = ['credit_card', 'credit_card', 'credit_card', 
                     'boleto', 'voucher', 'debit_card']
    
    for order_id in order_ids:
        num_payments = random.choices([1, 2], weights=[0.9, 0.1])[0]
        total_value = round(random.uniform(50, 1000), 2)
        
        for seq in range(1, num_payments + 1):
            payment_type = random.choice(payment_types)
            installments = random.randint(1, 12) if payment_type == 'credit_card' else 1
            value = total_value / num_payments
            
            cursor.execute("""
                INSERT INTO olist_order_payments VALUES (?, ?, ?, ?, ?)
            """, (order_id, seq, payment_type, installments, round(value, 2)))
    
    # Generate order reviews (800 - not all orders have reviews)
    print("  Generating order reviews...")
    reviewed_orders = random.sample(order_ids, 800)
    
    review_titles = [
        'Ótimo produto!', 'Muito bom', 'Recomendo', 'Chegou rápido',
        'Produto conforme descrito', 'Excelente', None, None, None,
        'Demorou para chegar', 'Produto veio com defeito', 'Não recomendo'
    ]
    
    review_messages = [
        'Produto de ótima qualidade, recomendo!',
        'Chegou dentro do prazo e bem embalado.',
        'Excelente custo-benefício.',
        'Superou minhas expectativas!',
        None,
        None,
        'Demorou mais do que o esperado.',
        'O produto não corresponde à descrição.',
        'Atendeu perfeitamente às minhas necessidades.',
    ]
    
    for order_id in reviewed_orders:
        review_id = generate_id(f"review_{order_id}")
        score = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.1, 0.3, 0.5])[0]
        
        title = random.choice(review_titles)
        if score <= 2:
            title = random.choice(['Não recomendo', 'Péssimo', None, 'Decepcionante'])
        
        message = random.choice(review_messages)
        creation = random_date()
        answer = creation + timedelta(days=random.randint(1, 7))
        
        cursor.execute("""
            INSERT INTO olist_order_reviews VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            review_id,
            order_id,
            score,
            title,
            message,
            creation.isoformat(),
            answer.isoformat(),
        ))
    
    # Create indexes for better query performance
    print("Creating indexes...")
    cursor.execute("CREATE INDEX idx_orders_customer ON olist_orders(customer_id)")
    cursor.execute("CREATE INDEX idx_order_items_order ON olist_order_items(order_id)")
    cursor.execute("CREATE INDEX idx_order_items_product ON olist_order_items(product_id)")
    cursor.execute("CREATE INDEX idx_order_items_seller ON olist_order_items(seller_id)")
    cursor.execute("CREATE INDEX idx_payments_order ON olist_order_payments(order_id)")
    cursor.execute("CREATE INDEX idx_reviews_order ON olist_order_reviews(order_id)")
    cursor.execute("CREATE INDEX idx_products_category ON olist_products(product_category_name)")
    
    conn.commit()
    
    # Print summary
    print("\n✓ Database created successfully!")
    print(f"\nDatabase path: {db_path}")
    print("\nTable summary:")
    
    for table in [
        'olist_customers', 'olist_sellers', 'olist_products', 'olist_orders',
        'olist_order_items', 'olist_order_payments', 'olist_order_reviews',
        'olist_geolocation', 'product_category_name_translation'
    ]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} rows")
    
    conn.close()
    
    return db_path


if __name__ == "__main__":
    setup_olist_database()
