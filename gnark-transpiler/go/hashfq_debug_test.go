package jolt_verifier

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/emulated"
	"github.com/consensys/gnark/test"
	"jolt_verifier/poseidon"
)

// DebugHashFqCircuit extracts intermediate state after round 0
type DebugHashFqCircuit struct {
	In1 frontend.Variable
	In2 frontend.Variable
	In3 frontend.Variable
	// Expected state after round 0 ARK+S-box (before MDS)
	Expected0AfterSbox frontend.Variable
	Expected1AfterSbox frontend.Variable
	Expected2AfterSbox frontend.Variable
	Expected3AfterSbox frontend.Variable
}

func (c *DebugHashFqCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Convert native Fr inputs to emulated Fq via bit decomposition
	e1 := fqField.FromBits(api.ToBinary(c.In1, 254)...)
	e2 := fqField.FromBits(api.ToBinary(c.In2, 254)...)
	e3 := fqField.FromBits(api.ToBinary(c.In3, 254)...)

	state := [4]*emulated.Element[emulated.BN254Fp]{fqField.Zero(), e1, e2, e3}

	// Round 0: ARK
	rc := poseidon.GetFqRoundConstants()
	for i := 0; i < 4; i++ {
		rcElem := fqField.NewElement(rc[i])
		state[i] = fqField.Add(state[i], rcElem)
	}

	// Round 0: S-box (x^5)
	for i := 0; i < 4; i++ {
		x := state[i]
		x2 := fqField.Mul(x, x)
		x4 := fqField.Mul(x2, x2)
		state[i] = fqField.Mul(x4, x)
	}

	// Convert back to Fr and assert
	bits0 := fqField.ToBits(state[0])
	bits1 := fqField.ToBits(state[1])
	bits2 := fqField.ToBits(state[2])
	bits3 := fqField.ToBits(state[3])

	result0 := api.FromBinary(bits0[:254]...)
	result1 := api.FromBinary(bits1[:254]...)
	result2 := api.FromBinary(bits2[:254]...)
	result3 := api.FromBinary(bits3[:254]...)

	api.AssertIsEqual(result0, c.Expected0AfterSbox)
	api.AssertIsEqual(result1, c.Expected1AfterSbox)
	api.AssertIsEqual(result2, c.Expected2AfterSbox)
	api.AssertIsEqual(result3, c.Expected3AfterSbox)

	return nil
}

// Compute expected state after round 0 ARK+S-box using big.Int
func computeStateAfterRound0Sbox(in1, in2, in3 *big.Int) [4]*big.Int {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
	rc := poseidon.GetFqRoundConstants()

	state := [4]*big.Int{new(big.Int), new(big.Int).Set(in1), new(big.Int).Set(in2), new(big.Int).Set(in3)}

	// ARK
	for i := 0; i < 4; i++ {
		state[i].Add(state[i], rc[i])
		state[i].Mod(state[i], fq)
	}

	// S-box: x^5
	exp5 := func(x *big.Int) *big.Int {
		x2 := new(big.Int).Mul(x, x)
		x2.Mod(x2, fq)
		x4 := new(big.Int).Mul(x2, x2)
		x4.Mod(x4, fq)
		r := new(big.Int).Mul(x4, x)
		r.Mod(r, fq)
		return r
	}

	for i := 0; i < 4; i++ {
		state[i] = exp5(state[i])
	}

	return state
}

func TestDebugHashFqRound0(t *testing.T) {
	in1 := big.NewInt(42)
	in2 := big.NewInt(123)
	in3 := big.NewInt(456)

	state := computeStateAfterRound0Sbox(in1, in2, in3)

	t.Logf("Expected state[0] after round 0 S-box: %s", state[0].String())
	t.Logf("Expected state[1] after round 0 S-box: %s", state[1].String())
	t.Logf("Expected state[2] after round 0 S-box: %s", state[2].String())
	t.Logf("Expected state[3] after round 0 S-box: %s", state[3].String())

	assignment := &DebugHashFqCircuit{
		In1:                in1,
		In2:                in2,
		In3:                in3,
		Expected0AfterSbox: state[0],
		Expected1AfterSbox: state[1],
		Expected2AfterSbox: state[2],
		Expected3AfterSbox: state[3],
	}

	var circuit DebugHashFqCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("DebugHashFqRound0 FAILED: %v", err)
	} else {
		t.Log("DebugHashFqRound0 PASSED - Circuit matches big.Int after ARK+Sbox")
	}
}

// DebugHashFqCircuitWithMDS extracts state after round 0 ARK+S-box+MDS
type DebugHashFqCircuitWithMDS struct {
	In1 frontend.Variable
	In2 frontend.Variable
	In3 frontend.Variable
	// Expected state after round 0 (ARK+S-box+MDS)
	Expected0AfterMDS frontend.Variable
	Expected1AfterMDS frontend.Variable
	Expected2AfterMDS frontend.Variable
	Expected3AfterMDS frontend.Variable
}

func (c *DebugHashFqCircuitWithMDS) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Convert native Fr inputs to emulated Fq via bit decomposition
	e1 := fqField.FromBits(api.ToBinary(c.In1, 254)...)
	e2 := fqField.FromBits(api.ToBinary(c.In2, 254)...)
	e3 := fqField.FromBits(api.ToBinary(c.In3, 254)...)

	state := [4]*emulated.Element[emulated.BN254Fp]{fqField.Zero(), e1, e2, e3}

	// Round 0: ARK
	rc := poseidon.GetFqRoundConstants()
	for i := 0; i < 4; i++ {
		rcElem := fqField.NewElement(rc[i])
		state[i] = fqField.Add(state[i], rcElem)
	}

	// Round 0: S-box (x^5)
	for i := 0; i < 4; i++ {
		x := state[i]
		x2 := fqField.Mul(x, x)
		x4 := fqField.Mul(x2, x2)
		state[i] = fqField.Mul(x4, x)
	}

	// Round 0: MDS
	mds := poseidon.GetFqMdsMatrix()
	var result [4]*emulated.Element[emulated.BN254Fp]
	for i := 0; i < 4; i++ {
		acc := fqField.Zero()
		for j := 0; j < 4; j++ {
			mdsElem := fqField.NewElement(mds[i][j])
			term := fqField.Mul(mdsElem, state[j])
			acc = fqField.Add(acc, term)
		}
		// FIX: Reduce after accumulation
		result[i] = fqField.Reduce(acc)
	}
	state = result

	// Convert back to Fr and assert (Reduce already done above)
	bits0 := fqField.ToBits(state[0])
	bits1 := fqField.ToBits(state[1])
	bits2 := fqField.ToBits(state[2])
	bits3 := fqField.ToBits(state[3])

	result0 := api.FromBinary(bits0[:254]...)
	result1 := api.FromBinary(bits1[:254]...)
	result2 := api.FromBinary(bits2[:254]...)
	result3 := api.FromBinary(bits3[:254]...)

	api.AssertIsEqual(result0, c.Expected0AfterMDS)
	api.AssertIsEqual(result1, c.Expected1AfterMDS)
	api.AssertIsEqual(result2, c.Expected2AfterMDS)
	api.AssertIsEqual(result3, c.Expected3AfterMDS)

	return nil
}

// Compute expected state after full round 0 (ARK+S-box+MDS) using big.Int
func computeStateAfterRound0Full(in1, in2, in3 *big.Int) [4]*big.Int {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
	rc := poseidon.GetFqRoundConstants()
	mds := poseidon.GetFqMdsMatrix()

	state := [4]*big.Int{new(big.Int), new(big.Int).Set(in1), new(big.Int).Set(in2), new(big.Int).Set(in3)}

	// ARK
	for i := 0; i < 4; i++ {
		state[i].Add(state[i], rc[i])
		state[i].Mod(state[i], fq)
	}

	// S-box: x^5
	exp5 := func(x *big.Int) *big.Int {
		x2 := new(big.Int).Mul(x, x)
		x2.Mod(x2, fq)
		x4 := new(big.Int).Mul(x2, x2)
		x4.Mod(x4, fq)
		r := new(big.Int).Mul(x4, x)
		r.Mod(r, fq)
		return r
	}

	for i := 0; i < 4; i++ {
		state[i] = exp5(state[i])
	}

	// MDS
	var result [4]*big.Int
	for i := 0; i < 4; i++ {
		acc := new(big.Int)
		for j := 0; j < 4; j++ {
			term := new(big.Int).Mul(mds[i][j], state[j])
			acc.Add(acc, term)
		}
		result[i] = acc.Mod(acc, fq)
	}

	return result
}

func TestDebugHashFqRound0Full(t *testing.T) {
	in1 := big.NewInt(42)
	in2 := big.NewInt(123)
	in3 := big.NewInt(456)

	state := computeStateAfterRound0Full(in1, in2, in3)

	t.Logf("Expected state[0] after round 0 full: %s", state[0].String())
	t.Logf("Expected state[1] after round 0 full: %s", state[1].String())
	t.Logf("Expected state[2] after round 0 full: %s", state[2].String())
	t.Logf("Expected state[3] after round 0 full: %s", state[3].String())

	assignment := &DebugHashFqCircuitWithMDS{
		In1:               in1,
		In2:               in2,
		In3:               in3,
		Expected0AfterMDS: state[0],
		Expected1AfterMDS: state[1],
		Expected2AfterMDS: state[2],
		Expected3AfterMDS: state[3],
	}

	var circuit DebugHashFqCircuitWithMDS
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("DebugHashFqRound0Full FAILED: %v", err)
	} else {
		t.Log("DebugHashFqRound0Full PASSED - Circuit matches big.Int after full round 0")
	}
}

// SimpleMDSCircuit tests just MDS multiplication with known inputs
type SimpleMDSCircuit struct {
	In0 frontend.Variable
	In1 frontend.Variable
	In2 frontend.Variable
	In3 frontend.Variable
	Out0 frontend.Variable
}

func (c *SimpleMDSCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	// Convert inputs
	e0 := fqField.FromBits(api.ToBinary(c.In0, 254)...)
	e1 := fqField.FromBits(api.ToBinary(c.In1, 254)...)
	e2 := fqField.FromBits(api.ToBinary(c.In2, 254)...)
	e3 := fqField.FromBits(api.ToBinary(c.In3, 254)...)
	state := [4]*emulated.Element[emulated.BN254Fp]{e0, e1, e2, e3}

	// MDS just for row 0
	mds := poseidon.GetFqMdsMatrix()
	acc := fqField.Zero()
	for j := 0; j < 4; j++ {
		mdsElem := fqField.NewElement(mds[0][j])
		term := fqField.Mul(mdsElem, state[j])
		acc = fqField.Add(acc, term)
	}

	bits := fqField.ToBits(acc)
	result := api.FromBinary(bits[:254]...)
	api.AssertIsEqual(result, c.Out0)
	return nil
}

func TestSimpleMDS(t *testing.T) {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
	mds := poseidon.GetFqMdsMatrix()

	// Simple inputs
	in0 := big.NewInt(1)
	in1 := big.NewInt(2)
	in2 := big.NewInt(3)
	in3 := big.NewInt(4)

	// Compute expected: out[0] = mds[0][0]*in0 + mds[0][1]*in1 + mds[0][2]*in2 + mds[0][3]*in3
	acc := new(big.Int)
	acc.Add(acc, new(big.Int).Mul(mds[0][0], in0))
	acc.Add(acc, new(big.Int).Mul(mds[0][1], in1))
	acc.Add(acc, new(big.Int).Mul(mds[0][2], in2))
	acc.Add(acc, new(big.Int).Mul(mds[0][3], in3))
	acc.Mod(acc, fq)

	t.Logf("MDS[0][0] = %s", mds[0][0].String())
	t.Logf("MDS[0][1] = %s", mds[0][1].String())
	t.Logf("MDS[0][2] = %s", mds[0][2].String())
	t.Logf("MDS[0][3] = %s", mds[0][3].String())
	t.Logf("in = [%s, %s, %s, %s]", in0, in1, in2, in3)
	t.Logf("Expected out[0] = %s", acc.String())

	assignment := &SimpleMDSCircuit{
		In0:  in0,
		In1:  in1,
		In2:  in2,
		In3:  in3,
		Out0: acc,
	}

	var circuit SimpleMDSCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("SimpleMDS FAILED: %v", err)
	} else {
		t.Log("SimpleMDS PASSED")
	}
}

// SingleMulCircuit tests a single emulated multiplication
type SingleMulCircuit struct {
	A frontend.Variable
	B frontend.Variable
	C frontend.Variable // expected result: A * B mod fq
}

func (c *SingleMulCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	a := fqField.FromBits(api.ToBinary(c.A, 254)...)
	b := fqField.FromBits(api.ToBinary(c.B, 254)...)

	result := fqField.Mul(a, b)

	bits := fqField.ToBits(result)
	r := api.FromBinary(bits[:254]...)
	api.AssertIsEqual(r, c.C)
	return nil
}

func TestSingleEmulatedMul(t *testing.T) {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)

	// Test: a * b where a and b are large enough to require reduction
	a, _ := new(big.Int).SetString("5472060717959818805561601436314318772174077789324455915672259473661306552146", 10) // mds[0][0]
	b := big.NewInt(3) // simple scalar

	expected := new(big.Int).Mul(a, b)
	expected.Mod(expected, fq)

	t.Logf("a = %s", a.String())
	t.Logf("b = %s", b.String())
	t.Logf("expected = a*b mod fq = %s", expected.String())

	assignment := &SingleMulCircuit{
		A: a,
		B: b,
		C: expected,
	}

	var circuit SingleMulCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("SingleEmulatedMul FAILED: %v", err)
	} else {
		t.Log("SingleEmulatedMul PASSED")
	}
}

// MulAddCircuit tests: mds[0][0]*1 + mds[0][1]*2 (just two terms)
type MulAddCircuit struct {
	In0 frontend.Variable
	In1 frontend.Variable
	Out frontend.Variable
}

func (c *MulAddCircuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	mds := poseidon.GetFqMdsMatrix()
	e0 := fqField.FromBits(api.ToBinary(c.In0, 254)...)
	e1 := fqField.FromBits(api.ToBinary(c.In1, 254)...)

	mds00 := fqField.NewElement(mds[0][0])
	mds01 := fqField.NewElement(mds[0][1])

	term0 := fqField.Mul(mds00, e0)
	term1 := fqField.Mul(mds01, e1)
	sum := fqField.Add(term0, term1)

	bits := fqField.ToBits(sum)
	r := api.FromBinary(bits[:254]...)
	api.AssertIsEqual(r, c.Out)
	return nil
}

func TestMulAddTwoTerms(t *testing.T) {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
	mds := poseidon.GetFqMdsMatrix()

	in0 := big.NewInt(1)
	in1 := big.NewInt(2)

	// mds[0][0]*1 + mds[0][1]*2
	term0 := new(big.Int).Mul(mds[0][0], in0)
	term1 := new(big.Int).Mul(mds[0][1], in1)
	expected := new(big.Int).Add(term0, term1)
	expected.Mod(expected, fq)

	t.Logf("mds[0][0]*1 = %s", term0.String())
	t.Logf("mds[0][1]*2 = %s", term1.String())
	t.Logf("Sum (before mod) = %s", new(big.Int).Add(term0, term1).String())
	t.Logf("Expected = %s", expected.String())

	assignment := &MulAddCircuit{
		In0: in0,
		In1: in1,
		Out: expected,
	}

	var circuit MulAddCircuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("MulAddTwoTerms FAILED: %v", err)
	} else {
		t.Log("MulAddTwoTerms PASSED")
	}
}

// MulAdd4Circuit tests: sum of 4 terms using accumulator loop
type MulAdd4Circuit struct {
	In0 frontend.Variable
	In1 frontend.Variable
	In2 frontend.Variable
	In3 frontend.Variable
	Out frontend.Variable
}

func (c *MulAdd4Circuit) Define(api frontend.API) error {
	fqField, err := emulated.NewField[emulated.BN254Fp](api)
	if err != nil {
		return err
	}

	mds := poseidon.GetFqMdsMatrix()
	inputs := []frontend.Variable{c.In0, c.In1, c.In2, c.In3}

	// Accumulate exactly like in SimpleMDSCircuit
	acc := fqField.Zero()
	for j := 0; j < 4; j++ {
		e := fqField.FromBits(api.ToBinary(inputs[j], 254)...)
		mdsElem := fqField.NewElement(mds[0][j])
		term := fqField.Mul(mdsElem, e)
		acc = fqField.Add(acc, term)
	}

	// TRY: Reduce before ToBits to ensure canonical form
	acc = fqField.Reduce(acc)

	bits := fqField.ToBits(acc)
	r := api.FromBinary(bits[:254]...)
	api.AssertIsEqual(r, c.Out)
	return nil
}

func TestMulAddFourTerms(t *testing.T) {
	fq, _ := new(big.Int).SetString("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10)
	mds := poseidon.GetFqMdsMatrix()

	in0 := big.NewInt(1)
	in1 := big.NewInt(2)
	in2 := big.NewInt(3)
	in3 := big.NewInt(4)

	// Compute sum of 4 terms
	acc := new(big.Int)
	for j := 0; j < 4; j++ {
		term := new(big.Int).Mul(mds[0][j], []*big.Int{in0, in1, in2, in3}[j])
		t.Logf("term[%d] = mds[0][%d] * in%d = %s", j, j, j, term.String())
		acc.Add(acc, term)
	}
	t.Logf("Sum before mod: %s", acc.String())
	acc.Mod(acc, fq)
	t.Logf("Expected (mod fq): %s", acc.String())

	assignment := &MulAdd4Circuit{
		In0: in0,
		In1: in1,
		In2: in2,
		In3: in3,
		Out: acc,
	}

	var circuit MulAdd4Circuit
	err := test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Errorf("MulAddFourTerms FAILED: %v", err)
	} else {
		t.Log("MulAddFourTerms PASSED")
	}
}
