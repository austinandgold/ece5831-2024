# main.py
import pricing


net_price = pricing.get_net_price(
    price=100,
    tax_rate=0.01
)

print(net_price)

import pricing as selling_price

net_price = selling_price.get_net_price(
    price=100,
    tax_rate=0.01
)

from pricing import get_net_price as calculate_net_price

net_price = calculate_net_price(
    price=100,
    tax_rate=0.1,
    discount=0.05
)
from pricing import *
from product import *

tax = get_tax(100)
print(tax)

import sys

for path in sys.path:
    print(path)

#import Sales.billing
import Sales.order
import Sales.delivery
import Sales.billing


Sales.order.create_sales_order()
Sales.delivery.create_delivery()
Sales.billing.create_billing()