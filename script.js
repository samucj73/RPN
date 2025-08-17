function calcularBF() {
  const sexo = document.getElementById("sexo").value;
  const altura = parseFloat(document.getElementById("altura").value);
  const pescoco = parseFloat(document.getElementById("pescoco").value);
  const cintura = parseFloat(document.getElementById("cintura").value);
  const quadril = parseFloat(document.getElementById("quadril").value);

  let bf;

  if (sexo === "M") {
    bf = 86.010 * Math.log10(cintura - pescoco) - 70.041 * Math.log10(altura) + 36.76;
  } else {
    if (isNaN(quadril)) {
      alert("Para mulheres é necessário informar o quadril!");
      return;
    }
    bf = 163.205 * Math.log10(cintura + quadril - pescoco) - 97.684 * Math.log10(altura) - 78.387;
  }

  bf = Math.round(bf * 100) / 100;

  let interpretacao = "";
  let posicao = 0;

  if (sexo === "M") {
    if (bf < 6) { interpretacao = `Abaixo do saudável (<6%)`; posicao = 5; }
    else if (bf <= 13) { interpretacao = `Atleta (6% a 13%)`; posicao = 20; }
    else if (bf <= 17) { interpretacao = `Fitness (14% a 17%)`; posicao = 40; }
    else if (bf <= 24) { interpretacao = `Normal (18% a 24%)`; posicao = 65; }
    else { interpretacao = `Obesidade (≥25%)`; posicao = 90; }
  } else {
    if (bf < 14) { interpretacao = `Abaixo do saudável (<14%)`; posicao = 5; }
    else if (bf <= 20) { interpretacao = `Atleta (14% a 20%)`; posicao = 20; }
    else if (bf <= 24) { interpretacao = `Fitness (21% a 24%)`; posicao = 40; }
    else if (bf <= 31) { interpretacao = `Normal (25% a 31%)`; posicao = 65; }
    else { interpretacao = `Obesidade (≥32%)`; posicao = 90; }
  }

  document.getElementById("resultado").innerHTML =
    `<b>Seu percentual de gordura corporal é:</b> ${bf}% <br><br> <b>Classificação:</b> ${interpretacao}`;

  document.getElementById("seta").style.left = posicao + "%";
}
