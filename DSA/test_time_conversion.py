def convert_time(s):
    """Convert 12-hour time format to 24-hour format"""
    minute = s[3]+s[4]
    sec = s[6]+s[7]
    
    # Extract hour part
    hour_str = s[0]+s[1]
    hour = int(hour_str)
    
    # Handle AM/PM conversion
    if s[-2]=='A':  # AM
        if hour == 12:
            hour = 0  # 12:xx:xx AM becomes 00:xx:xx
        # For other AM times, hour stays the same
    elif s[-2]=='P':  # PM
        if hour != 12:
            hour = hour + 12  # Add 12 for PM times except 12:xx:xx PM
        # 12:xx:xx PM stays 12:xx:xx
    
    # Format hour with zero-padding
    formatted_hour = f"{hour:02d}"
    return formatted_hour+':'+minute+':'+sec

# Test cases
test_cases = [
    ("12:00:00AM", "00:00:00"),  # Midnight
    ("12:00:00PM", "12:00:00"),  # Noon
    ("01:00:00AM", "01:00:00"),  # 1 AM
    ("01:00:00PM", "13:00:00"),  # 1 PM
    ("11:59:59AM", "11:59:59"),  # 11:59:59 AM
    ("11:59:59PM", "23:59:59"),  # 11:59:59 PM
    ("07:05:45PM", "19:05:45"),  # Original test case
    ("12:30:15AM", "00:30:15"),  # 12:30:15 AM
    ("12:30:15PM", "12:30:15"),  # 12:30:15 PM
]

print("Testing time conversion:")
print("=" * 40)
all_passed = True

for input_time, expected in test_cases:
    result = convert_time(input_time)
    status = "✓ PASS" if result == expected else "✗ FAIL"
    if result != expected:
        all_passed = False
    print(f"{input_time:12} -> {result:8} (expected: {expected:8}) {status}")

print("=" * 40)
if all_passed:
    print("All test cases passed! ✓")
else:
    print("Some test cases failed! ✗")
