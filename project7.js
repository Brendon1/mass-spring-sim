// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.
function GetModelViewMatrix( translationX, translationY, translationZ, rotationX, rotationY )
{
	var M_RotX = [
		1,  0, 					 0,					  0,
		0,  Math.cos(rotationX), Math.sin(rotationX), 0,
		0, -Math.sin(rotationX), Math.cos(rotationX), 0,
		0,  0, 					 0,					  1
	];

	var M_RotY = [
		Math.cos(rotationY), 0, -Math.sin(rotationY), 0,
		0,					 1,  0, 				  0,
		Math.sin(rotationY), 0,  Math.cos(rotationY), 0,
		0, 				 	 0,  0, 				  1
	];

	// transformation matrix.
	var trans = [
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translationX, translationY, translationZ, 1
	];

	var rotation = MatrixMult(M_RotX, M_RotY)
	var mv = MatrixMult(trans, rotation);

	return mv;
}

class MeshDrawer
{
	// The constructor is a good place for taking care of the necessary initializations.
	constructor()
	{
		this.M_Identity = [1, 0, 0, 0,
						   0, 1, 0, 0,
						   0, 0, 1, 0,
						   0, 0, 0, 1];

		this.M_SwapYZ = [1, 0, 0, 0,
						 0, 0, 1, 0,
						 0, 1, 0, 0,
						 0, 0, 0, 1];

		// Specify the shader program by giving it custom vertex and fragment shader GLSL code
		this.prog = InitShaderProgram(textureVS, textureFS);

		// Get locations of attributes
		this.pos = gl.getAttribLocation( this.prog, 'pos');
		this.txc = gl.getAttribLocation( this.prog, 'txc');
		this.norms = gl.getAttribLocation( this.prog, 'norms');

		// Get locations of uniform variables
		this.mvp = gl.getUniformLocation(this.prog, 'mvp');
		this.mv = gl.getUniformLocation(this.prog, 'mv');
		this.matNorm = gl.getUniformLocation(this.prog, 'matNorm');
		this.axisTrans = gl.getUniformLocation(this.prog, 'axisTrans');
		this.sampler = gl.getUniformLocation(this.prog, 'tex');
		this.showTex = gl.getUniformLocation(this.prog, 'showTex');
		this.lightDir = gl.getUniformLocation(this.prog, 'lightDir');
		this.shininess = gl.getUniformLocation(this.prog, 'shininess');

		// By default don't swap the vertical axis, nor render the texture 
		gl.useProgram(this.prog);
		gl.uniformMatrix4fv(this.axisTrans, false, new Float32Array(this.M_Identity));
		gl.uniform1i(this.showTex, false);

		// Create texture and buffers
		this.vertexPosBuffer = gl.createBuffer();
		this.textureCoordBuffer = gl.createBuffer();
		this.normalsBuffer = gl.createBuffer();

		// The number of triangles in the mesh to be drawn
		this.numTriangles = 0;
	}
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions,
	// an array of 2D texture coordinates, and an array of vertex normals.
	// Every item in these arrays is a floating point value, representing one
	// coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex
	// position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array
	// form the texture coordinate of a vertex and every three consecutive 
	// elements in the normals array form a vertex normal.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords, normals )
	{
		// Update the contents of the vertex buffer objects.
		this.numTriangles = vertPos.length / 3;

		// Fill buffer with vertex positions of users obj file
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexPosBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertPos), gl.STATIC_DRAW);

		// Fill buffer with texture coordinates of users texture file
		gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);

		// Fill buffer with vertex normals of users obj file
		gl.bindBuffer(gl.ARRAY_BUFFER, this.normalsBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
	}
	
	// This method is called when the user changes the state of the
	// "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		// Set the uniform parameter(s) of the vertex shader
		gl.useProgram(this.prog);

		if(swap)
			gl.uniformMatrix4fv(this.axisTrans, false, new Float32Array(this.M_SwapYZ));
		else
			gl.uniformMatrix4fv(this.axisTrans, false, new Float32Array(this.M_Identity));
	}
	
	// This method is called to draw the triangular mesh.
	// The arguments are the model-view-projection transformation matrixMVP,
	// the model-view transformation matrixMV, the same matrix returned
	// by the GetModelViewProjection function above, and the normal
	// transformation matrix, which is the inverse-transpose of matrixMV.
	draw( matrixMVP, matrixMV, matrixNormal )
	{
		// specify program
		gl.useProgram(this.prog);
		// pass the model-view-projection matrix to vertex shader uniform variable
		gl.uniformMatrix4fv(this.mvp, false, matrixMVP);
		gl.uniformMatrix4fv(this.mv, false, matrixMV);
		gl.uniformMatrix3fv(this.matNorm, false, matrixNormal);

		// Bind the buffer to vertex positions
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexPosBuffer);
		// Specify how webGL should interpret the data
		gl.vertexAttribPointer(this.pos, 3, gl.FLOAT, false, 0, 0);
		// enable the attribute
		gl.enableVertexAttribArray(this.pos);

		// Bind the buffer for texture coordinates
		gl.bindBuffer(gl.ARRAY_BUFFER, this.textureCoordBuffer);
		// Specify how webGL should interpret the data
		gl.vertexAttribPointer(this.txc, 2, gl.FLOAT, false, 0, 0);
		// enable the attribute
		gl.enableVertexAttribArray(this.txc);

		// Bind the buffer for the vertex normals
		gl.bindBuffer(gl.ARRAY_BUFFER, this.normalsBuffer);
		// Specify how webGL should interpret the data
		gl.vertexAttribPointer(this.norms, 3, gl.FLOAT, false, 0, 0);
		// enable the attribute
		gl.enableVertexAttribArray(this.norms);

		// Complete the WebGL initializations before drawing
		gl.drawArrays( gl.TRIANGLES, 0, this.numTriangles );
	}
	
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	setTexture( img )
	{
		// Create a new texture
		const mytex = gl.createTexture();
		// Bind the texture
		gl.bindTexture(gl.TEXTURE_2D, mytex);

		// You can set the texture image data using the following command.
		gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img );

		// generate the mipmaps for the texture
		gl.generateMipmap(gl.TEXTURE_2D);

		// Now that we have a texture, it might be a good idea to set
		// some uniform parameter(s) of the fragment shader, so that it uses the texture.
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

		// Use GPU texture unit 0
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, mytex);

		// Specify that the texture will be sampled from texture unit 0
		gl.useProgram(this.prog);
		gl.uniform1i(this.sampler, 0);

		// Set the texture rendering to the selected option in the UI
		this.showTexture(document.getElementById("show-texture").checked);
	}
	
	// This method is called when the user changes the state of the
	// "Show Texture" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	showTexture( show )
	{
		// set the uniform parameter(s) of the fragment shader to specify if it should use the texture.
		gl.useProgram(this.prog);
		gl.uniform1i(this.showTex, show);
	}
	
	// This method is called to set the incoming light direction
	setLightDir( x, y, z )
	{
		// set the uniform parameter(s) of the fragment shader to specify the light direction.
		gl.useProgram(this.prog);
		gl.uniform3f(this.lightDir, x, y, z);
	}
	
	// This method is called to set the shininess of the material
	setShininess( shininess )
	{
		// set the uniform parameter(s) of the fragment shader to specify the shininess.
		gl.useProgram(this.prog);
		gl.uniform1f(this.shininess, shininess);
	}
}

// Vertex Shader
var textureVS = `
	attribute vec3 pos;
	attribute vec2 txc;
	attribute vec3 norms;

	uniform mat4 mvp;
	uniform mat4 mv;
	uniform mat3 matNorm;
	uniform mat4 axisTrans;

	varying vec2 texCoord;
	varying vec3 vertNormInterp;
	varying vec3 vertPos;

	void main()
	{
		gl_Position = mvp * axisTrans * vec4(pos, 1.0);
		vertNormInterp = matNorm * norms;
		vertPos = vec3(mv * axisTrans * vec4(pos, 1.0));
		texCoord = txc;
	}
`;

// Fragment Shader
var textureFS = `
	precision mediump float;
	uniform bool showTex;
	uniform sampler2D tex;
	uniform float shininess;
	uniform vec3 lightDir;

	varying vec2 texCoord;
	varying vec3 vertNormInterp;
	varying vec3 vertPos;
	void main()
	{
		vec3 normal = normalize(vertNormInterp);
		vec3 lightColor = vec3(1.0, 1.0, 1.0);
		float lightIntensity = 1.0;
		vec3 diffuseColor = vec3(1.0, 1.0, 1.0);
		vec3 specColor = vec3(1.0, 1.0, 1.0);

		vec3 lightDirNorm = normalize(lightDir);

		// Clamp theta to be non-negative
		float lambert = max(dot(lightDirNorm, normal), 0.0);
		float specular = 0.0;

		if(lambert > 0.0){
			vec3 viewDir = normalize(-vertPos);

			vec3 h = normalize(lightDirNorm + viewDir);
			float cosPhi = max(dot(h, normal), 0.0);
			specular = pow(cosPhi, shininess);
		}
		
		vec3 textureColor = vec3(1.0, 1.0, 1.0);
		if(showTex){
			textureColor = texture2D(tex, texCoord).rgb;
		}

		textureColor = lambert * textureColor;
		vec3 specularColor = specular * specColor;
		vec3 color = textureColor + specularColor;
		
		gl_FragColor = vec4(color, 1.0);
	}
`;

// This function is called for every step of the simulation.
// Its job is to advance the simulation for the given time step duration dt.
// It updates the given positions and velocities.
function SimTimeStep( dt, positions, velocities, springs, stiffness, damping, particleMass, gravity, restitution )
{
	// Compute the total force of each particle

	// List of total force per particle, initialized with gravitational force
	var forces = Array( positions.length ).fill(gravity.mul(particleMass));

	// Compute mass spring force for each particle
	for(let i = 0; i < springs.length; i++){
		// The particles at both ends of the spring
		let p0 = positions[springs[i].p0];
		let p1 = positions[springs[i].p1];

		// Velocities of those particles
		let pv0 = velocities[springs[i].p0];
		let pv1 = velocities[springs[i].p1];

		let springLen = (p1.sub(p0)).len(); // Length of spring
		let rest = springs[i].rest; // Resting length of spring
		let dir = (p1.sub(p0)).div(springLen); // Direction of spring

		// Spring force on p0 and p1
		let Fs0 = dir.mul(stiffness * (springLen - rest));
		let Fs1 = Fs0.mul(-1);
		
		// Damping force on p0 and p1
		let Fd0 = dir.mul(damping * (pv1.sub(pv0)).dot(dir));
		let Fd1 = Fd0.mul(-1);

		// Total mass spring forces on p0 and p1
		let F0 = Fs0.add(Fd0);
		let F1 = Fs1.add(Fd1);

		// Add the mass spring forces to system of forces
		forces[springs[i].p0] = forces[springs[i].p0].add(F0);
		forces[springs[i].p1] = forces[springs[i].p1].add(F1);
	}
	
	// Update positions and velocities
	for(let i = 0; i < positions.length; i++){
		acceleration = forces[i].div(particleMass);

		velocities[i].inc(acceleration.mul(dt));
		positions[i].inc(velocities[i].mul(dt));
	}
	
	// Handle collisions
	for(let i = 0; i < positions.length; i++){
		for(axis of ['x','y','z']){
			if(positions[i][axis] > 1){
				let n = positions[i][axis] - 1;
				let m = restitution * n;
				positions[i][axis] -= (n + m);

				velocities[i][axis] *= (-restitution);
			}
			else if(positions[i][axis] < -1){
				let n = positions[i][axis] + 1;
				let m = restitution * n;
				positions[i][axis] -= (n + m);

				velocities[i][axis] *= (-restitution);
			}
		}
	}
	
}

