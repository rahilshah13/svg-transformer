import java.nio.file._, StandardWatchEventKinds._
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters._
import scala.sys.process._
import java.io.FileOutputStream
import java.util.concurrent.{Executors, TimeUnit, TimeoutException}

@main def potraceDaemon(inDir: String, outDir: String): Unit =
  val (in, out) = (Paths.get(inDir), Paths.get(outDir))
  val watcher = in.getFileSystem.newWatchService()
  in.register(watcher, ENTRY_CREATE)
  Files.createDirectories(out)
  
  println(s"[*] Edge detection daemon started. Watching: $in -> Output: $out")
  
  // Dedicated single-thread executor for interruptible decoding
  val imageExecutor = Executors.newSingleThreadExecutor()
  
  def processFile(file: Path): Unit =
    if Files.isRegularFile(file) && !file.getFileName.toString.startsWith(".") then
      println(s"[*] Processing: ${file.getFileName}...")
      
      var lastSize = -1L
      var currentSize = try Files.size(file) catch case _ => -1L
      var checks = 0
      while currentSize != lastSize && checks < 20 do
        lastSize = currentSize
        Thread.sleep(250)
        currentSize = try Files.size(file) catch case _ => -1L
        checks += 1

      val decodeTask = imageExecutor.submit(() => Option(ImageIO.read(file.toFile)))

      try
        val imgOpt = decodeTask.get(5, TimeUnit.SECONDS)
        imgOpt match
          case Some(img) =>
            val (w, h) = (img.getWidth, img.getHeight)
            val pixels = img.getRGB(0, 0, w, h, null, 0, w)
            
            val pbm = Files.createTempFile("edge", ".pbm")
            val outStream = new FileOutputStream(pbm.toFile)
            try
              outStream.write(s"P4\n$w $h\n".getBytes("US-ASCII"))
              for y <- 0 until h do
                var byteVal = 0
                var bitCount = 0
                val rowOffset = y * w
                for x <- 0 until w do
                  val gx = if x > 0 && x < w - 1 then (pixels(rowOffset + x + 1) & 0xff) - (pixels(rowOffset + x - 1) & 0xff) else 0
                  val gy = if y > 0 && y < h - 1 then (pixels((y + 1) * w + x) & 0xff) - (pixels((y - 1) * w + x) & 0xff) else 0
                  val bit = if math.hypot(gx, gy) > 60 then 1 else 0
                  byteVal = (byteVal << 1) | bit
                  bitCount += 1
                  if bitCount == 8 then
                    outStream.write(byteVal)
                    byteVal = 0
                    bitCount = 0
                if bitCount > 0 then
                  outStream.write(byteVal << (8 - bitCount))
            finally outStream.close()
            
            val svg = Files.createTempFile("edge", ".svg")
            s"potrace -s -o ${svg.toAbsolutePath} ${pbm.toAbsolutePath}".!
            
            val outFile = out.resolve(s"${file.getFileName}.svg")
            Files.copy(svg, outFile, StandardCopyOption.REPLACE_EXISTING)
            Files.deleteIfExists(pbm); Files.deleteIfExists(svg)
            println(s"[+] Successfully generated SVG: ${file.getFileName}.svg")
            
          case None =>
            println(s"[!] Warning: ImageIO returned null for ${file.getFileName}. Skipping.")
      catch
        case _: TimeoutException =>
          decodeTask.cancel(true)
          println(s"[!] Error: Image decoding timed out for ${file.getFileName}.")
        case e: Exception =>
          println(s"[!] Error processing ${file.getFileName}: ${e.getMessage}")

  // Process existing files on startup
  try
    val stream = Files.newDirectoryStream(in)
    try for f <- stream.asScala do processFile(f)
    finally stream.close()
  catch case e: Exception => println(s"[!] Startup directory scan failed: ${e.getMessage}")

  // Watch loop
  while true do
    val key = watcher.take()
    try
      for e <- key.pollEvents().asScala if e.kind() == ENTRY_CREATE do
        val file = in.resolve(e.context.asInstanceOf[Path])
        processFile(file)
    finally
      key.reset()