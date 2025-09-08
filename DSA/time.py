s="07:05:45PM"
time=s.split(":")
print(time)
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
last = formatted_hour+':'+minute+':'+sec
print(last)