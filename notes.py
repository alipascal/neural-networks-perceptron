# ----------------------------
def test_display():
	X = (0, 1)
	z = 1

	wx1z = 5
	wx2z = -3
	wbz = 1

	w = [wx1z, wx2z]


	netz = np.dot(X,w) + wbz
	oz = sigmoid(netz)
	tz = z
	pred_z = seuil(netz)

	color_valid = "grey"
	if pred_z == tz:
		color_valid = "green"
		print(f"le point {X} est bien placé")
	else:
		color_valid = "red"
		print(f"le point {X} est mal placé")

	dz = (tz - oz) * oz * (1 - oz)

	print(f"point{X}, netz={netz}, oz={oz}, pred_z={pred_z}, tz={tz}, dz={dz}")

	# x = list(range(2))
	# y = [np.dot((xi, 1), w) + wbz for xi in x]
	x = np.linspace(-2, 2, 400)
	y = (wx1z * x + wbz) / -wx2z

	fig, ax = plt.subplots()
	ax.set_aspect('equal', adjustable='box')

	ax.plot(x, y)


	# Selector
	plt.scatter(X[0], X[1], alpha=0.5, color=color_valid, s=200)

	# Points 
	plt.scatter(0, 1, color="black", marker='+')
	plt.scatter(0, 0, color="black", marker="_")
	plt.scatter(1, 0, color="black", marker="+")
	plt.scatter(1, 1, color="black", marker="+")
	plt.show()

# Arrow
# xp, yp = 5, np.dot((5, 1), w) + wbz # Point pour la flèche
# dx, dy = 1, (np.dot((6, 1), w) + wbz) - yp # Vecteur directeur de la droite
# # Vecteur normal (perpendiculaire)
# normal_x, normal_y = -dy, dx
# normal_x, normal_y = normal_x / np.hypot(normal_x, normal_y), normal_y / np.hypot(normal_x, normal_y)
# # Dessin de la flèche perpendiculaire
# ax.arrow(xp, yp, normal_x, normal_y, head_width=1, head_length=1, fc='r', ec='r', label="Flèche perpendiculaire")
# ax.legend()



# ax.set_xlim(min(x), max(x))
# ax.set_ylim(min(y), max(y))


data = [ 
	[5.1,3.5],
	[4.9,3.0],
	[4.7,3.2],
	[4.6,3.1],
	[5.0,3.6],
	[5.4,3.9],
	[4.6,3.4],
	[5.0,3.4],
	[4.4,2.9],
	[4.9,3.1],
	[5.4,3.7],
	[4.8,3.4],
	[4.8,3.0],
	[4.3,3.0],
	[5.8,4.0],
	[5.7,4.4],
	[5.4,3.9],
	[5.1,3.5],
	[5.7,3.8],
	[5.1,3.8],
	[5.4,3.4],
	[5.1,3.7],
	[4.6,3.6],
	[5.1,3.3],
	[4.8,3.4],
	[5.0,3.0],
	[5.0,3.4],
	[5.2,3.5],
	[5.2,3.4],
	[4.7,3.2],
	[4.8,3.1],
	[5.4,3.4],
	[5.2,4.1],
	[5.5,4.2],
	[4.9,3.1],
	[5.0,3.2],
	[5.5,3.5],
	[4.9,3.6],
	[4.4,3.0],
	[5.1,3.4],
	[5.0,3.5],
	[4.5,2.3],
	[4.4,3.2],
	[5.0,3.5],
]