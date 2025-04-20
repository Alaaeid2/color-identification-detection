import cv2
import pandas as pd

# Load color data
csv_path = 'colors.csv'
colors = pd.read_csv(csv_path)

# Load image
img_path = 'OIP.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (800, 600))  # resize for ease

clicked = False
r = g = b = xpos = ypos = 0

# Get closest color name
def get_color_name(R, G, B):
    min_dist = float('inf')
    cname = "Unknown"
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

# Mouse callback function
def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        xpos, ypos = x, y
        b, g, r = img[y, x]
        b, g, r = int(b), int(g), int(r)

cv2.namedWindow('Color Identifier')
cv2.setMouseCallback('Color Identifier', draw_function)

while True:
    cv2.imshow("Color Identifier", img)
    if clicked:
        cv2.rectangle(img, (20, 20), (600, 60), (b, g, r), -1)
        text = get_color_name(r, g, b) + f' R={r} G={g} B={b}'
        cv2.putText(img, text, (30, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if r + g + b >= 600:
            cv2.putText(img, text, (30, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    if cv2.waitKey(20) & 0xFF == 27:  # press ESC to exit
        break

cv2.destroyAllWindows()